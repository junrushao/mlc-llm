/*!
 *  Copyright (c) 2023 by Contributors
 * \file llm_chat.cc
 * \brief Implementation of llm chat.
 */
#define PICOJSON_USE_INT64
#define __STDC_FORMAT_MACROS

#include "llm_chat.h"

#include <picojson.h>
#include <tokenizers_cpp.h>
#include <tvm/ir/attrs.h>
#include <tvm/runtime/container/optional.h>
#include <tvm/runtime/disco/session.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/relax_vm/memory_manager.h>

#include <cctype>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <list>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <unordered_set>

#include "conversation.h"

namespace mlc {
namespace llm {

using tvm::Device;
using namespace tvm::runtime;

//----------------------------
// Tokenizers
//----------------------------
using tokenizers::Tokenizer;

std::string LoadBytesFromFile(const std::string& path) {
  std::ifstream fs(path, std::ios::in | std::ios::binary);
  ICHECK(!fs.fail()) << "Cannot open " << path;
  std::string data;
  fs.seekg(0, std::ios::end);
  size_t size = static_cast<size_t>(fs.tellg());
  fs.seekg(0, std::ios::beg);
  data.resize(size);
  fs.read(data.data(), size);
  return data;
}

std::unique_ptr<Tokenizer> TokenizerFromPath(const std::string& _path) {
  std::filesystem::path path(_path);
  std::filesystem::path sentencepiece;
  std::filesystem::path huggingface;
  std::filesystem::path rwkvworld;
  CHECK(std::filesystem::exists(path)) << "Cannot find tokenizer via path: " << _path;
  if (std::filesystem::is_directory(path)) {
    sentencepiece = path / "tokenizer.model";
    huggingface = path / "tokenizer.json";
    rwkvworld = path / "tokenizer_model";
    // Check ByteLevelBPE
    {
      std::filesystem::path merges_path = path / "merges.txt";
      std::filesystem::path vocab_path = path / "vocab.json";
      std::filesystem::path added_tokens_path = path / "added_tokens.json";
      if (std::filesystem::exists(merges_path) && std::filesystem::exists(vocab_path) &&
          std::filesystem::exists(added_tokens_path)) {
        std::string vocab = LoadBytesFromFile(vocab_path.string());
        std::string merges = LoadBytesFromFile(merges_path.string());
        std::string added_tokens = LoadBytesFromFile(added_tokens_path.string());
        return Tokenizer::FromBlobByteLevelBPE(vocab, merges, added_tokens);
      }
    }
  } else {
    sentencepiece = path.parent_path() / "tokenizer.model";
    huggingface = path.parent_path() / "tokenizer.json";
    rwkvworld = path.parent_path() / "tokenizer_model";
  }
  if (std::filesystem::exists(sentencepiece)) {
    return Tokenizer::FromBlobSentencePiece(LoadBytesFromFile(sentencepiece.string()));
  }
  if (std::filesystem::exists(huggingface)) {
    return Tokenizer::FromBlobJSON(LoadBytesFromFile(huggingface.string()));
  }
  if (std::filesystem::exists(rwkvworld)) {
    return Tokenizer::FromBlobRWKVWorld(rwkvworld.string());
  }
  LOG(FATAL) << "Cannot find any tokenizer under: " << _path;
}

//------------------------------
// support functions
//------------------------------
inline size_t FindEffectiveUTF8Pos(const std::string& s) {
  int pos = s.size() - 1;
  for (; pos >= 0; pos--) {
    if ((s[pos] & 0x80) == 0x00) {
      return pos + 1;
    } else if (pos - 1 >= 0 && (s[pos - 1] & 0xE0) == 0xC0 && (s[pos] & 0xC0) == 0x80) {
      return pos + 1;
    } else if (pos - 2 >= 0 && (s[pos - 2] & 0xF0) == 0xE0 && (s[pos - 1] & 0xC0) == 0x80 &&
               (s[pos] & 0xC0) == 0x80) {
      return pos + 1;
    } else if (pos - 3 >= 0 && (s[pos - 3] & 0xF8) == 0xF0 && (s[pos - 2] & 0xC0) == 0x80 &&
               (s[pos - 1] & 0xC0) == 0x80 && (s[pos] & 0xC0) == 0x80) {
      return pos + 1;
    }
  }
  return pos + 1;
}

inline std::string Concat(const std::vector<std::string>& inputs) {
  std::ostringstream os;
  for (const auto& x : inputs) {
    os << x;
  }
  return os.str();
}

//------------------------------
// Chat module
//------------------------------
class LLMChatModule;

/*!
 * \brief Implements the chat conversation wrapper
 */
class LLMChat {
  friend class LLMChatModule;

 public:
  explicit LLMChat(DLDevice device) : device_(device) {}

  /*!
   * \return Text describing runtime stats.
   */
  std::string RuntimeStatsText() {
    std::ostringstream os;
    os << "prefill: " << std::setprecision(1) << std::fixed
       << this->prefill_total_tokens / (this->prefill_total_time + this->embed_total_time)
       << " tok/s"
       << ", decode: " << std::setprecision(1) << std::fixed
       << this->decode_total_tokens / this->decode_total_time << " tok/s";
    return os.str();
  }

  /*!
   * \brief Load JSON config and override options.
   * \param config_json A json config in picojson type that is partially specifies
   *        some of the options.
   * \param partial_update Whether it's a partial update or full update, if set to true,
   *        we perform a partial update on some of the provided options; if set to false, all
   *        options must be provided.
   * \note This function overrides existing configurations.
   */
  void LoadJSONOverride(const picojson::value& config_json, bool partial_update = false) {
    picojson::object config = config_json.get<picojson::object>();
    if (config.count("temperature")) {
      CHECK(config["temperature"].is<double>());
      this->temperature_ = config["temperature"].get<double>();
    } else {
      CHECK(partial_update) << "Key \"temperature\" not found.";
    }
    if (config.count("vocab_size")) {
      CHECK(config["vocab_size"].is<int64_t>());
      this->vocab_size_ = config["vocab_size"].get<int64_t>();
    } else {
      CHECK(partial_update) << "Key \"vocab_size\" not found.";
    }
    if (config.count("max_window_size")) {
      CHECK(config["max_window_size"].is<int64_t>());
      this->max_window_size_ = config["max_window_size"].get<int64_t>();
    } else {
      CHECK(partial_update) << "Key \"max_window_size\" not found.";
    }
    if (config.count("model_name")) {
      CHECK(config["model_name"].is<std::string>());
      this->model_name_ = config["model_name"].get<std::string>();
    } else {
      CHECK(partial_update) << "Key \"model_name\" not found.";
    }
    if (config.count("repetition_penalty")) {
      CHECK(config["repetition_penalty"].is<double>());
      CHECK(this->repetition_penalty_ > 0) << "Repetition penalty must be a positive number!";
      this->repetition_penalty_ = config["repetition_penalty"].get<double>();
    } else {
      CHECK(partial_update) << "Key \"repetition_penalty\" not found.";
    }
    if (config.count("top_p")) {
      CHECK(config["top_p"].is<double>());
      this->top_p_ = config["top_p"].get<double>();
    } else {
      CHECK(partial_update) << "Key \"top_p\" not found.";
    }
    if (config.count("mean_gen_len")) {
      CHECK(config["mean_gen_len"].is<int64_t>());
      this->mean_gen_len_ = config["mean_gen_len"].get<int64_t>();
    } else {
      CHECK(partial_update) << "Key \"mean_gen_len\" not found.";
    }
    // NOTE: for backward compact
    // max gen len is optional
    if (config.count("max_gen_len")) {
      CHECK(config["max_gen_len"].is<int64_t>());
      this->max_gen_len_ = config["max_gen_len"].get<int64_t>();
    }
    if (config.count("shift_fill_factor")) {
      CHECK(config["shift_fill_factor"].is<double>());
      this->shift_fill_factor_ = config["shift_fill_factor"].get<double>();
    } else {
      CHECK(partial_update) << "Key \"shift_fill_factor\" not found.";
    }
    if (config.count("conv_template")) {
      ICHECK(config["conv_template"].is<std::string>());
      std::string conv_template = config["conv_template"].get<std::string>();
      this->conversation_ = Conversation::FromTemplate(conv_template);
      if (config.count("conv_config")) {
        // conv_config can override conv_template
        this->conversation_.LoadJSONOverride(config["conv_config"], true);
      }
    } else if (config.count("conv_config")) {
      // without conv template, conv_config needs to be a complete config
      this->conversation_.LoadJSONOverride(config["conv_config"], false);
    } else {
      CHECK(partial_update) << "Key \"conv_template\" and \"conv_config\" not found.";
    }
  }

  /*!
   * \brief Load JSON config and override options.
   * \param config_str A json config string that partially specifies some of the options.
   * \param partial_update Whether it's a partial update or full update, if set to true,
   *        we perform a partial update on some of the provided options; if set to false, all
   *        options must be provided.
   * \note This function overrides existing configurations.
   */
  void LoadJSONOverride(const std::string& config_str, bool partial_update = false) {
    picojson::value config_json;
    std::string err = picojson::parse(config_json, config_str);
    if (!err.empty()) {
      LOG(FATAL) << err;
      return;
    }
    LoadJSONOverride(config_json, partial_update);
  }

  std::string GetConfigJSON() const { return SerializeConfigToJSONValue().serialize(true); }

  /*!
   * \brief Reload model, tokenizers and configurations from the specified model path.
   * \param executable The module to reload.
   * \param model_path The path to search for models.
   * \param app_config_json The JSON string used to partially override the configuration loaded from
   * disk, default to empty string.
   */
  void Reload(String lib_path, String model_path, String app_config_json = "") {
    // Step 1. Set tokenizer.
    this->tokenizer_ = TokenizerFromPath(model_path);

    // Step 2. Initialize vm, we use the packed function mechanism
    // so there is no explicit abi dependency on these extra
    // classes other than basic tvm runtime.
    tvm::runtime::Module executable = tvm::runtime::Module::LoadFromFile(lib_path);
    auto f_func_exists_ = executable->GetFunction("func_exists");
    ICHECK(f_func_exists_.defined()) << "TVM runtime cannot find func_exists";

    auto fthreaded_session = tvm::runtime::Registry::Get("runtime.disco.SessionThreaded");
    ICHECK(fthreaded_session) << "TVM runtime cannot find runtime.disco.SessionThreaded";
    session_ = (*fthreaded_session)(2);
    // session_ = (*fthreaded_session)(1);

    auto fnccl_init = session_->GetGlobalFunc("runtime.disco.nccl.init_ccl");
    session_->CallPacked(fnccl_init, 0, 1);
    // session_->CallPacked(fnccl_init, 0);

    auto fvm_load_module = session_->GetGlobalFunc("runtime.disco.load_vm_module");
    auto mod = session_->CallPacked(fvm_load_module, lib_path, Device{DLDeviceType(0), 0});

    auto fmodule_get_function = session_->GetGlobalFunc("runtime.ModuleGetFunction");

    prefill_func_ = session_->CallPacked(fmodule_get_function, mod, "prefill", false);
    LOG(INFO) << "prefill id " << prefill_func_->reg_id;
    func_exists_[prefill_func_] = f_func_exists_("prefill");
    embed_func_ = session_->CallPacked(fmodule_get_function, mod, "embed", false);
    LOG(INFO) << "embed id " << embed_func_->reg_id;
    func_exists_[embed_func_] = f_func_exists_("embed");
    prefill_with_embed_func_ =
        session_->CallPacked(fmodule_get_function, mod, "prefill_with_embed", false);
    LOG(INFO) << "prefill_with_embed id " << prefill_with_embed_func_->reg_id;
    func_exists_[prefill_with_embed_func_] = f_func_exists_("prefill_with_embed");
    decode_func_ = session_->CallPacked(fmodule_get_function, mod, "decode", false);
    LOG(INFO) << "decode id " << decode_func_->reg_id;
    func_exists_[decode_func_] = f_func_exists_("decode");
    encoding_without_cache_func_ =
        session_->CallPacked(fmodule_get_function, mod, "encoding_without_cache", false);
    LOG(INFO) << "encoding_without_cache id " << encoding_without_cache_func_->reg_id;
    func_exists_[encoding_without_cache_func_] = f_func_exists_("encoding_without_cache");
    softmax_func_ =
        session_->CallPacked(fmodule_get_function, mod, "softmax_with_temperature", false);
    LOG(INFO) << "softmax id " << softmax_func_->reg_id;
    func_exists_[softmax_func_] = f_func_exists_("softmax_with_temperature");
    tuple_getitem_func_ = session_->GetGlobalFunc("vm.builtin.tuple_getitem");
    LOG(INFO) << "tuple_getitem id " << tuple_getitem_func_->reg_id;
    auto fsample_topp_from_prob_ptr =
        tvm::runtime::Registry::Get("vm.builtin.sample_top_p_from_prob");
    ICHECK(fsample_topp_from_prob_ptr)
        << "Cannot find env function vm.builtin.sample_top_p_from_prob";
    fsample_topp_from_prob_ = *fsample_topp_from_prob_ptr;
    auto fsample_topp_from_logits_ptr =
        tvm::runtime::Registry::Get("vm.builtin.sample_top_p_from_logits");
    ICHECK(fsample_topp_from_logits_ptr)
        << "Cannot find env function vm.builtin.sample_top_p_from_logits";
    fsample_topp_from_logits_ = *fsample_topp_from_logits_ptr;

    // Step 3. Load params in nd-array cache.

    auto fcreate_shard_loader = session_->GetGlobalFunc("runtime.disco.ShardLoader");
    LOG(INFO) << "create shard loader id " << fcreate_shard_loader->reg_id;
    auto fload_shard = session_->GetGlobalFunc("runtime.disco.ShardLoaderLoadAll");
    LOG(INFO) << "load shard id " << fload_shard->reg_id;
    // FIXME: fix the shard info
    std::string shard_info_str =
        "{ \"param_2\": 0 ,  \"param_3\": 0 ,  \"param_4\": 0 ,  \"param_5\": 0 ,  \"param_6\": 0 "
        ",  \"param_7\": 0 ,  \"param_8\": 1 ,  \"param_9\": 1 ,  \"param_10\": 0 ,  \"param_11\": "
        "0 ,  \"param_14\": 0 ,  \"param_15\": 0 ,  \"param_12\": 1 ,  \"param_13\": 1 ,  "
        "\"param_18\": 0 ,  \"param_19\": 0 ,  \"param_20\": 0 ,  \"param_21\": 0 ,  \"param_22\": "
        "0 ,  \"param_23\": 0 ,  \"param_24\": 1 ,  \"param_25\": 1 ,  \"param_26\": 0 ,  "
        "\"param_27\": 0 ,  \"param_30\": 0 ,  \"param_31\": 0 ,  \"param_28\": 1 ,  \"param_29\": "
        "1 ,  \"param_34\": 0 ,  \"param_35\": 0 ,  \"param_36\": 0 ,  \"param_37\": 0 ,  "
        "\"param_38\": 0 ,  \"param_39\": 0 ,  \"param_40\": 1 ,  \"param_41\": 1 ,  \"param_42\": "
        "0 ,  \"param_43\": 0 ,  \"param_46\": 0 ,  \"param_47\": 0 ,  \"param_44\": 1 ,  "
        "\"param_45\": 1 ,  \"param_50\": 0 ,  \"param_51\": 0 ,  \"param_52\": 0 ,  \"param_53\": "
        "0 ,  \"param_54\": 0 ,  \"param_55\": 0 ,  \"param_56\": 1 ,  \"param_57\": 1 ,  "
        "\"param_58\": 0 ,  \"param_59\": 0 ,  \"param_62\": 0 ,  \"param_63\": 0 ,  \"param_60\": "
        "1 ,  \"param_61\": 1 ,  \"param_66\": 0 ,  \"param_67\": 0 ,  \"param_68\": 0 ,  "
        "\"param_69\": 0 ,  \"param_70\": 0 ,  \"param_71\": 0 ,  \"param_72\": 1 ,  \"param_73\": "
        "1 ,  \"param_74\": 0 ,  \"param_75\": 0 ,  \"param_78\": 0 ,  \"param_79\": 0 ,  "
        "\"param_76\": 1 ,  \"param_77\": 1 ,  \"param_82\": 0 ,  \"param_83\": 0 ,  \"param_84\": "
        "0 ,  \"param_85\": 0 ,  \"param_86\": 0 ,  \"param_87\": 0 ,  \"param_88\": 1 ,  "
        "\"param_89\": 1 ,  \"param_90\": 0 ,  \"param_91\": 0 ,  \"param_94\": 0 ,  \"param_95\": "
        "0 ,  \"param_92\": 1 ,  \"param_93\": 1 ,  \"param_98\": 0 ,  \"param_99\": 0 ,  "
        "\"param_100\": 0 ,  \"param_101\": 0 ,  \"param_102\": 0 ,  \"param_103\": 0 ,  "
        "\"param_104\": 1 ,  \"param_105\": 1 ,  \"param_106\": 0 ,  \"param_107\": 0 ,  "
        "\"param_110\": 0 ,  \"param_111\": 0 ,  \"param_108\": 1 ,  \"param_109\": 1 ,  "
        "\"param_114\": 0 ,  \"param_115\": 0 ,  \"param_116\": 0 ,  \"param_117\": 0 ,  "
        "\"param_118\": 0 ,  \"param_119\": 0 ,  \"param_120\": 1 ,  \"param_121\": 1 ,  "
        "\"param_122\": 0 ,  \"param_123\": 0 ,  \"param_126\": 0 ,  \"param_127\": 0 ,  "
        "\"param_124\": 1 ,  \"param_125\": 1 ,  \"param_130\": 0 ,  \"param_131\": 0 ,  "
        "\"param_132\": 0 ,  \"param_133\": 0 ,  \"param_134\": 0 ,  \"param_135\": 0 ,  "
        "\"param_136\": 1 ,  \"param_137\": 1 ,  \"param_138\": 0 ,  \"param_139\": 0 ,  "
        "\"param_142\": 0 ,  \"param_143\": 0 ,  \"param_140\": 1 ,  \"param_141\": 1 ,  "
        "\"param_146\": 0 ,  \"param_147\": 0 ,  \"param_148\": 0 ,  \"param_149\": 0 ,  "
        "\"param_150\": 0 ,  \"param_151\": 0 ,  \"param_152\": 1 ,  \"param_153\": 1 ,  "
        "\"param_154\": 0 ,  \"param_155\": 0 ,  \"param_158\": 0 ,  \"param_159\": 0 ,  "
        "\"param_156\": 1 ,  \"param_157\": 1 ,  \"param_162\": 0 ,  \"param_163\": 0 ,  "
        "\"param_164\": 0 ,  \"param_165\": 0 ,  \"param_166\": 0 ,  \"param_167\": 0 ,  "
        "\"param_168\": 1 ,  \"param_169\": 1 ,  \"param_170\": 0 ,  \"param_171\": 0 ,  "
        "\"param_174\": 0 ,  \"param_175\": 0 ,  \"param_172\": 1 ,  \"param_173\": 1 ,  "
        "\"param_178\": 0 ,  \"param_179\": 0 ,  \"param_180\": 0 ,  \"param_181\": 0 ,  "
        "\"param_182\": 0 ,  \"param_183\": 0 ,  \"param_184\": 1 ,  \"param_185\": 1 ,  "
        "\"param_186\": 0 ,  \"param_187\": 0 ,  \"param_190\": 0 ,  \"param_191\": 0 ,  "
        "\"param_188\": 1 ,  \"param_189\": 1 ,  \"param_194\": 0 ,  \"param_195\": 0 ,  "
        "\"param_196\": 0 ,  \"param_197\": 0 ,  \"param_198\": 0 ,  \"param_199\": 0 ,  "
        "\"param_200\": 1 ,  \"param_201\": 1 ,  \"param_202\": 0 ,  \"param_203\": 0 ,  "
        "\"param_206\": 0 ,  \"param_207\": 0 ,  \"param_204\": 1 ,  \"param_205\": 1 ,  "
        "\"param_210\": 0 ,  \"param_211\": 0 ,  \"param_212\": 0 ,  \"param_213\": 0 ,  "
        "\"param_214\": 0 ,  \"param_215\": 0 ,  \"param_216\": 1 ,  \"param_217\": 1 ,  "
        "\"param_218\": 0 ,  \"param_219\": 0 ,  \"param_222\": 0 ,  \"param_223\": 0 ,  "
        "\"param_220\": 1 ,  \"param_221\": 1 ,  \"param_226\": 0 ,  \"param_227\": 0 ,  "
        "\"param_228\": 0 ,  \"param_229\": 0 ,  \"param_230\": 0 ,  \"param_231\": 0 ,  "
        "\"param_232\": 1 ,  \"param_233\": 1 ,  \"param_234\": 0 ,  \"param_235\": 0 ,  "
        "\"param_238\": 0 ,  \"param_239\": 0 ,  \"param_236\": 1 ,  \"param_237\": 1 ,  "
        "\"param_242\": 0 ,  \"param_243\": 0 ,  \"param_244\": 0 ,  \"param_245\": 0 ,  "
        "\"param_246\": 0 ,  \"param_247\": 0 ,  \"param_248\": 1 ,  \"param_249\": 1 ,  "
        "\"param_250\": 0 ,  \"param_251\": 0 ,  \"param_254\": 0 ,  \"param_255\": 0 ,  "
        "\"param_252\": 1 ,  \"param_253\": 1 ,  \"param_258\": 0 ,  \"param_259\": 0 ,  "
        "\"param_260\": 0 ,  \"param_261\": 0 ,  \"param_262\": 0 ,  \"param_263\": 0 ,  "
        "\"param_264\": 1 ,  \"param_265\": 1 ,  \"param_266\": 0 ,  \"param_267\": 0 ,  "
        "\"param_270\": 0 ,  \"param_271\": 0 ,  \"param_268\": 1 ,  \"param_269\": 1 ,  "
        "\"param_274\": 0 ,  \"param_275\": 0 ,  \"param_276\": 0 ,  \"param_277\": 0 ,  "
        "\"param_278\": 0 ,  \"param_279\": 0 ,  \"param_280\": 1 ,  \"param_281\": 1 ,  "
        "\"param_282\": 0 ,  \"param_283\": 0 ,  \"param_286\": 0 ,  \"param_287\": 0 ,  "
        "\"param_284\": 1 ,  \"param_285\": 1 ,  \"param_290\": 0 ,  \"param_291\": 0 ,  "
        "\"param_292\": 0 ,  \"param_293\": 0 ,  \"param_294\": 0 ,  \"param_295\": 0 ,  "
        "\"param_296\": 1 ,  \"param_297\": 1 ,  \"param_298\": 0 ,  \"param_299\": 0 ,  "
        "\"param_302\": 0 ,  \"param_303\": 0 ,  \"param_300\": 1 ,  \"param_301\": 1 ,  "
        "\"param_306\": 0 ,  \"param_307\": 0 ,  \"param_308\": 0 ,  \"param_309\": 0 ,  "
        "\"param_310\": 0 ,  \"param_311\": 0 ,  \"param_312\": 1 ,  \"param_313\": 1 ,  "
        "\"param_314\": 0 ,  \"param_315\": 0 ,  \"param_318\": 0 ,  \"param_319\": 0 ,  "
        "\"param_316\": 1 ,  \"param_317\": 1 ,  \"param_322\": 0 ,  \"param_323\": 0 ,  "
        "\"param_324\": 0 ,  \"param_325\": 0 ,  \"param_326\": 0 ,  \"param_327\": 0 ,  "
        "\"param_328\": 1 ,  \"param_329\": 1 ,  \"param_330\": 0 ,  \"param_331\": 0 ,  "
        "\"param_334\": 0 ,  \"param_335\": 0 ,  \"param_332\": 1 ,  \"param_333\": 1 ,  "
        "\"param_338\": 0 ,  \"param_339\": 0 ,  \"param_340\": 0 ,  \"param_341\": 0 ,  "
        "\"param_342\": 0 ,  \"param_343\": 0 ,  \"param_344\": 1 ,  \"param_345\": 1 ,  "
        "\"param_346\": 0 ,  \"param_347\": 0 ,  \"param_350\": 0 ,  \"param_351\": 0 ,  "
        "\"param_348\": 1 ,  \"param_349\": 1 ,  \"param_354\": 0 ,  \"param_355\": 0 ,  "
        "\"param_356\": 0 ,  \"param_357\": 0 ,  \"param_358\": 0 ,  \"param_359\": 0 ,  "
        "\"param_360\": 1 ,  \"param_361\": 1 ,  \"param_362\": 0 ,  \"param_363\": 0 ,  "
        "\"param_366\": 0 ,  \"param_367\": 0 ,  \"param_364\": 1 ,  \"param_365\": 1 ,  "
        "\"param_370\": 0 ,  \"param_371\": 0 ,  \"param_372\": 0 ,  \"param_373\": 0 ,  "
        "\"param_374\": 0 ,  \"param_375\": 0 ,  \"param_376\": 1 ,  \"param_377\": 1 ,  "
        "\"param_378\": 0 ,  \"param_379\": 0 ,  \"param_382\": 0 ,  \"param_383\": 0 ,  "
        "\"param_380\": 1 ,  \"param_381\": 1 ,  \"param_386\": 0 ,  \"param_387\": 0 ,  "
        "\"param_388\": 0 ,  \"param_389\": 0 ,  \"param_390\": 0 ,  \"param_391\": 0 ,  "
        "\"param_392\": 1 ,  \"param_393\": 1 ,  \"param_394\": 0 ,  \"param_395\": 0 ,  "
        "\"param_398\": 0 ,  \"param_399\": 0 ,  \"param_396\": 1 ,  \"param_397\": 1 ,  "
        "\"param_402\": 0 ,  \"param_403\": 0 ,  \"param_404\": 0 ,  \"param_405\": 0 ,  "
        "\"param_406\": 0 ,  \"param_407\": 0 ,  \"param_408\": 1 ,  \"param_409\": 1 ,  "
        "\"param_410\": 0 ,  \"param_411\": 0 ,  \"param_414\": 0 ,  \"param_415\": 0 ,  "
        "\"param_412\": 1 ,  \"param_413\": 1 ,  \"param_418\": 0 ,  \"param_419\": 0 ,  "
        "\"param_420\": 0 ,  \"param_421\": 0 ,  \"param_422\": 0 ,  \"param_423\": 0 ,  "
        "\"param_424\": 1 ,  \"param_425\": 1 ,  \"param_426\": 0 ,  \"param_427\": 0 ,  "
        "\"param_430\": 0 ,  \"param_431\": 0 ,  \"param_428\": 1 ,  \"param_429\": 1 ,  "
        "\"param_434\": 0 ,  \"param_435\": 0 ,  \"param_436\": 0 ,  \"param_437\": 0 ,  "
        "\"param_438\": 0 ,  \"param_439\": 0 ,  \"param_440\": 1 ,  \"param_441\": 1 ,  "
        "\"param_442\": 0 ,  \"param_443\": 0 ,  \"param_446\": 0 ,  \"param_447\": 0 ,  "
        "\"param_444\": 1 ,  \"param_445\": 1 ,  \"param_450\": 0 ,  \"param_451\": 0 ,  "
        "\"param_452\": 0 ,  \"param_453\": 0 ,  \"param_454\": 0 ,  \"param_455\": 0 ,  "
        "\"param_456\": 1 ,  \"param_457\": 1 ,  \"param_458\": 0 ,  \"param_459\": 0 ,  "
        "\"param_462\": 0 ,  \"param_463\": 0 ,  \"param_460\": 1 ,  \"param_461\": 1 ,  "
        "\"param_466\": 0 ,  \"param_467\": 0 ,  \"param_468\": 0 ,  \"param_469\": 0 ,  "
        "\"param_470\": 0 ,  \"param_471\": 0 ,  \"param_472\": 1 ,  \"param_473\": 1 ,  "
        "\"param_474\": 0 ,  \"param_475\": 0 ,  \"param_478\": 0 ,  \"param_479\": 0 ,  "
        "\"param_476\": 1 ,  \"param_477\": 1 ,  \"param_482\": 0 ,  \"param_483\": 0 ,  "
        "\"param_484\": 0 ,  \"param_485\": 0 ,  \"param_486\": 0 ,  \"param_487\": 0 ,  "
        "\"param_488\": 1 ,  \"param_489\": 1 ,  \"param_490\": 0 ,  \"param_491\": 0 ,  "
        "\"param_494\": 0 ,  \"param_495\": 0 ,  \"param_492\": 1 ,  \"param_493\": 1 ,  "
        "\"param_498\": 0 ,  \"param_499\": 0 ,  \"param_500\": 0 ,  \"param_501\": 0 ,  "
        "\"param_502\": 0 ,  \"param_503\": 0 ,  \"param_504\": 1 ,  \"param_505\": 1 ,  "
        "\"param_506\": 0 ,  \"param_507\": 0 ,  \"param_510\": 0 ,  \"param_511\": 0 ,  "
        "\"param_508\": 1 ,  \"param_509\": 1}";
    // std::string shard_info_str = "{}";
    std::string json_path = model_path + "/ndarray-cache.json";
    auto ndarray_cache_metadata = LoadBytesFromFile(json_path);
    PackedFunc tmp(nullptr);
    auto loader = session_->CallPacked(fcreate_shard_loader, json_path, ndarray_cache_metadata,
                                       shard_info_str, tmp);
    params_ = session_->CallPacked(fload_shard, loader);
    // Step 4. KV cache creation.

    auto fcreate_kv_cache =
        session_->CallPacked(fmodule_get_function, mod, "create_kv_cache", false);
    kv_cache_ = session_->CallPacked(fcreate_kv_cache);

    fkvcache_array_popn_ = session_->GetGlobalFunc("vm.builtin.attention_kv_cache_array_popn");
    // Step 5. KV cache reset.
    reset_kv_cache_func_ = session_->CallPacked(fmodule_get_function, mod, "reset_kv_cache", false);
    if (!f_func_exists_("reset_kv_cache")) {
      auto attention_kv_cache_array_clear_ptr =
          session_->GetGlobalFunc("vm.builtin.attention_kv_cache_array_clear");

      reset_kv_cache_func_ = attention_kv_cache_array_clear_ptr;
      support_backtracking_kv_ = true;
    } else {
      // if there is a customized reset kv
      // then it may not be the typical transformer model
      // and we disable backtracking kv feature
      support_backtracking_kv_ = false;
    }

    // Step 6. Process config json string.
    std::ifstream config_istream((model_path + "/mlc-chat-config.json").c_str());
    std::ostringstream config_ostream;
    ICHECK(config_istream);
    config_ostream << config_istream.rdbuf();
    std::string config_str = config_ostream.str();
    LoadJSONOverride(config_str, false);

    // Step 7. Process metadata

    // fixme: load from json
    this->model_name_ = "Llama-2-7b-chat-hf-q4f16_1";
    this->max_window_size_ = 2048;

    // Step 8. Override configuration from app_config_json.
    if (!app_config_json.empty()) {
      LoadJSONOverride(app_config_json, true);
    }

    this->ResetChat();
  }

  void ResetChat() {
    // TODO(mlc-team): add conversation_.Reset to preserve system prompt
    // and initial message.
    // this->conversation_ = Conversation::Create(this->conversation_.conv_template);
    this->conversation_.Reset();
    this->ResetRuntimeStats();
    this->ResetKVCache();
    this->total_seq_len_ = 0;
  }

  /*! \brief reset the runtime stats. */
  void ResetRuntimeStats() {
    this->prefill_total_tokens = 0;
    this->decode_total_tokens = 0;
    this->embed_total_time = 0;
    this->prefill_total_time = 0;
    this->decode_total_time = 0;
    this->sample_total_time = 0;
  }

  static std::string GetConcatPrompt(const std::vector<std::string>& prompt_array,
                                     size_t prefix_end, size_t suffix_start) {
    std::ostringstream os;
    for (size_t i = 0; i < prefix_end; ++i) {
      os << prompt_array[i];
    }
    for (size_t i = suffix_start; i < prompt_array.size(); ++i) {
      os << prompt_array[i];
    }
    return os.str();
  }

  /**
   * \brief Get input tokens based on history
   * \param place_in_prompt The place of the input message in the prompt.
   */
  std::vector<int32_t> GetInputTokens(PlaceInPrompt place_in_prompt = PlaceInPrompt::kAll) {
    std::vector<int32_t> tokens;
    std::vector<std::string> prompts;

    if (this->total_seq_len_ == 0) {
      prompts = this->conversation_.GetPromptArray(place_in_prompt);
      if (this->conversation_.add_bos) {
        tokens.insert(tokens.begin(), bos_token_id_);
      }
      if (this->conversation_.prefix_tokens.size() != 0) {
        tokens.insert(tokens.begin(), this->conversation_.prefix_tokens.begin(),
                      this->conversation_.prefix_tokens.end());
      }
    } else {
      prompts = this->conversation_.GetPromptArrayLastRound(place_in_prompt);
    }
    // first try to encode all
    std::string all_prompt = GetConcatPrompt(prompts, 0, 0);
    std::vector<int32_t> encoded = this->tokenizer_->Encode(all_prompt);
    tokens.insert(tokens.end(), encoded.begin(), encoded.end());
    if (this->total_seq_len_ + tokens.size() + this->mean_gen_len_ < this->max_window_size_) {
      return tokens;
    }
    // need shift window and re-encode
    this->total_seq_len_ = 0;
    this->ResetKVCache();
    tokens.clear();
    if (this->conversation_.add_bos) {
      tokens.insert(tokens.begin(), bos_token_id_);
    }
    if (this->conversation_.prefix_tokens.size() != 0) {
      tokens.insert(tokens.begin(), this->conversation_.prefix_tokens.begin(),
                    this->conversation_.prefix_tokens.end());
    }
    std::vector<std::string> all_prompts = this->conversation_.GetPromptArray();
    // get estimate of the fragment
    size_t ctx_length = this->tokenizer_->Encode(all_prompts[0]).size();
    size_t start_re_encode_pos = 0;
    for (int i = all_prompts.size() - 1; i > 0; --i) {
      ctx_length += this->tokenizer_->Encode(all_prompts[i]).size();
      if (ctx_length >= this->shift_fill_factor_ * this->max_window_size_ &&
          i + 2 < all_prompts.size()) {
        start_re_encode_pos = i;
        break;
      }
    }
    // keep system
    if (this->conversation_.system.empty()) {
      all_prompt = GetConcatPrompt(all_prompts, 0, start_re_encode_pos);
    } else {
      all_prompt = GetConcatPrompt(all_prompts, 1, start_re_encode_pos);
    }
    encoded = this->tokenizer_->Encode(all_prompt);
    tokens.insert(tokens.end(), encoded.begin(), encoded.end());
    if (tokens.size() >= this->max_window_size_) {
      LOG(WARNING)
          << "The prompt tokens are more than `max_window_size`, the input will be truncated.";
      ICHECK_GT(this->max_window_size_, this->mean_gen_len_);
      std::vector<int32_t> truncated_tokens(
          tokens.end() - (this->max_window_size_ - this->mean_gen_len_), tokens.end());
      return truncated_tokens;
    } else if (tokens.size() + this->mean_gen_len_ >= this->max_window_size_) {
      LOG(WARNING)
          << "The prompt tokens are too long and the generated text may be incomplete, due to "
             "limited `max_window_size`. ";
    }
    return tokens;
  }

  // get statically allocated input token
  NDArray GetInputTokenNDArray(const std::vector<int32_t>& token_ids) {
    // try realloc
    if (!input_token_ids_.defined()) {
      int64_t init_size = 2048;
      while (init_size < static_cast<int64_t>(token_ids.size())) {
        init_size *= 2;
      }
      input_token_ids_ = NDArray::Empty({1, init_size}, DataType::Int(32), device_);
    } else {
      int64_t init_size = input_token_ids_->shape[1];
      while (init_size < static_cast<int64_t>(token_ids.size())) {
        init_size *= 2;
      }
      if (init_size != input_token_ids_->shape[1]) {
        input_token_ids_ = NDArray::Empty({1, init_size}, DataType::Int(32), device_);
      }
    }
    ICHECK_LE(token_ids.size(), input_token_ids_->shape[1]) << "Input tokens exceed window size";
    NDArray view = input_token_ids_.CreateView(
        ShapeTuple({1, static_cast<int64_t>(token_ids.size())}), input_token_ids_->dtype);
    if (token_ids.size() > 0) {
      view.CopyFromBytes(token_ids.data(), token_ids.size() * sizeof(int32_t));
    }
    return view;
  }

  std::vector<int32_t> PrepareBeforeEmbedding(std::string inp, bool append_conversation = true,
                                              PlaceInPrompt place_in_prompt = PlaceInPrompt::kAll) {
    if (conversation_.separator_style == SeparatorStyle::kLM ||
        conversation_.separator_style == SeparatorStyle::kCodeCompletion) {
      this->ResetChat();
    }
    if (reset_stats_per_prefill_) {
      this->ResetRuntimeStats();
    }
    output_ids_.clear();
    appeared_token_ids_.clear();
    output_message_.clear();
    stop_triggered_ = false;
    if (append_conversation) {
      conversation_.AppendMessage(conversation_.roles[0], inp);
      conversation_.AppendReplyHeader(conversation_.roles[1]);
    }

    return this->GetInputTokens(place_in_prompt);
  }

  DRef CopyToWorker0(const NDArray& host_array) {
    Device dev{DLDeviceType(0), 0};
    auto func = session_->GetGlobalFunc("runtime.disco.empty");
    ShapeTuple shape = host_array.Shape();
    DataType dtype = host_array.DataType();
    DRef dref = session_->CallPacked(func, shape, dtype, dev);
    session_->CopyToWorker0(host_array, dref);
    return dref;
  }

  /*!
   * \brief Given the text input, generate the embedding of the tokenized prompt.
   * \param inp The input text string.
   * \param append_conversation Whether to append the input message to conversation.
   * \param place_in_prompt The place of the input message in the prompt.
   * \return the embedding of the tokenized prompt.
   */
  DRef EmbedStep(std::string inp, bool append_conversation = true,
                 PlaceInPrompt place_in_prompt = PlaceInPrompt::kAll) {
    std::vector<int32_t> prompt_tokens =
        PrepareBeforeEmbedding(inp, append_conversation, place_in_prompt);
    int64_t token_len = static_cast<int64_t>(prompt_tokens.size());
    if (token_len == 0) {
      auto empty_func = session_->GetGlobalFunc("runtime.disco.empty");
      return session_->CallPacked(empty_func, ShapeTuple({}), DataType::Float(32),
                                  Device{DLDeviceType(0), 0});
    }

    CHECK(func_exists_[embed_func_])
        << "In order to use the embedding functionality, make sure you "
           "build the model in MLC-LLM with `sep_embed` option on.";
    auto tstart = std::chrono::high_resolution_clock::now();

    NDArray input_data = this->GetInputTokenNDArray(prompt_tokens);
    DRef input_dref = CopyToWorker0(input_data);
    DRef embedding = session_->CallPacked(embed_func_, input_dref, params_);

    int32_t new_seq_len = total_seq_len_ + token_len;
    total_seq_len_ = new_seq_len;

    auto tend = std::chrono::high_resolution_clock::now();

    this->embed_total_time += static_cast<double>((tend - tstart).count()) / 1e9;

    return embedding;
  }

  /*!
   * \brief Prefill given embeddings. Can optionally decode the output next token.
   * \param embedding The embedding to prefill with.
   * \param decode_next_token Whether to decode next token.
   */
  void PrefillWithEmbedStep(DRef embedding, bool decode_next_token = true) {
    LOG(FATAL) << "disco is not compatible with embed step";
    NDArray embedding_ndarray;
    session_->CopyFromWorker0(embedding_ndarray, embedding);
    session_->SyncWorker(0);
    if (embedding_ndarray.Shape().size() == 0) {
      return;
    }

    auto tstart = std::chrono::high_resolution_clock::now();

    int64_t token_len = embedding_ndarray.Shape()[1];
    NDArray logits_on_device = this->ForwardEmbeddings(embedding, total_seq_len_);

    if (!decode_next_token) {
      auto tend = std::chrono::high_resolution_clock::now();
      this->prefill_total_time += static_cast<double>((tend - tstart).count()) / 1e9;
      this->prefill_total_tokens += token_len;
      return;
    }

    int32_t next_token = this->SampleTokenFromLogits(logits_on_device, temperature_, top_p_);

    auto tend = std::chrono::high_resolution_clock::now();

    this->prefill_total_time += static_cast<double>((tend - tstart).count()) / 1e9;
    this->prefill_total_tokens += token_len;
    this->ProcessNextToken(next_token);
  }

  /*!
   * \brief Generate the next token given a prompt. Can optionally decode the output next token.
   * \param inp The input text string.
   * \param append_conversation Whether to append the input message to conversation.
   * \param decode_next_token Whether to decode next token.
   * \param place_in_prompt The place of the input message in the prompt.
   */
  void PrefillStep(std::string inp, bool append_conversation = true, bool decode_next_token = true,
                   PlaceInPrompt place_in_prompt = PlaceInPrompt::kAll) {
    if (func_exists_[embed_func_] && func_exists_[prefill_with_embed_func_]) {
      // Temporarily placed inside `PrefillStep` for compatibility in transition.
      // Will be separated out in the future.
      DRef embedding = EmbedStep(inp, append_conversation, place_in_prompt);
      PrefillWithEmbedStep(embedding, decode_next_token);
      return;
    }
    std::vector<int32_t> prompt_tokens =
        this->PrepareBeforeEmbedding(inp, append_conversation, place_in_prompt);
    int64_t token_len = static_cast<int64_t>(prompt_tokens.size());
    if (token_len == 0) return;

    auto tstart = std::chrono::high_resolution_clock::now();

    int32_t new_seq_len = total_seq_len_ + token_len;
    NDArray logits_on_device = this->ForwardTokens(prompt_tokens, new_seq_len);
    // this->UpdateLogitsOrProbOnCPUSync(logits_on_device);
    // TVMSynchronize(device_.device_type, device_.device_id, nullptr);
    // // print first few logits for eyeballs
    // std::ostringstream os;
    // for (int i = 0; i < 10; ++i) {
    //   if (i != 0) os << ", ";
    //   os << static_cast<float*>(logits_on_cpu_->data)[i];
    // }
    // LOG(INFO) << "logits[:10] =[" << os.str() << "]";
    total_seq_len_ = new_seq_len;

    if (!decode_next_token) {
      auto tend = std::chrono::high_resolution_clock::now();
      this->prefill_total_time += static_cast<double>((tend - tstart).count()) / 1e9;
      this->prefill_total_tokens += token_len;
      return;
    }

    int32_t next_token = this->SampleTokenFromLogits(logits_on_device, temperature_, top_p_);

    auto tend = std::chrono::high_resolution_clock::now();

    this->prefill_total_time += static_cast<double>((tend - tstart).count()) / 1e9;
    this->prefill_total_tokens += token_len;
    this->ProcessNextToken(next_token);
  }

  void DecodeStep() {
    ICHECK(!output_ids_.empty());
    int32_t last_token = output_ids_.back();
    tvm::runtime::NDArray input_data = GetInputTokenNDArray({last_token});

    auto tstart = std::chrono::high_resolution_clock::now();

    NDArray logits_on_device = this->ForwardTokens({last_token}, total_seq_len_ + 1);
    total_seq_len_ += 1;

    int32_t next_token = this->SampleTokenFromLogits(logits_on_device, temperature_, top_p_);

    auto tend = std::chrono::high_resolution_clock::now();

    this->decode_total_time += static_cast<double>((tend - tstart).count()) / 1e9;
    this->decode_total_tokens += 1;
    this->ProcessNextToken(next_token);
  }

  bool Stopped() { return stop_triggered_; }

  std::string GetMessage() {
    // remove non-utf8 characters
    size_t effective_end = FindEffectiveUTF8Pos(output_message_);
    while (effective_end > 0 && output_message_[effective_end - 1] == '\n') {
      --effective_end;
    }
    size_t effective_begin = 0;
    while (effective_begin < effective_end && output_message_[effective_begin] == ' ') {
      ++effective_begin;
    }
    std::string cropped_message =
        output_message_.substr(effective_begin, effective_end - effective_begin);
    return cropped_message;
  }

  // do some quick evaluation of the pipeline
  void Evaluate(int64_t token_len, int64_t generate_len) {
    this->ResetKVCache();
    std::vector<int32_t> tokens;
    for (int i = 0; i < token_len - 1; ++i) {
      tokens.push_back(2);
    }
    tokens.insert(tokens.begin(), bos_token_id_);

    std::vector<int32_t> first_sample_data = {6234};

    // warm up: skip first run
    this->ForwardTokens(tokens, token_len);
    this->ForwardTokens(first_sample_data, token_len + 1);
    this->ResetKVCache();

    // encoding
    auto encoding_start = std::chrono::high_resolution_clock::now();
    this->ForwardTokens(tokens, token_len);
    TVMSynchronize(device_.device_type, device_.device_id, nullptr);
    auto encoding_end = std::chrono::high_resolution_clock::now();
    double encoding_ms = static_cast<double>((encoding_end - encoding_start).count()) / 1e6;
    LOG(INFO) << "encoding-time=" << encoding_ms << "ms, ";

    double decoding_ms_total = 0;
    // start encoding
    for (int i = 0; i < generate_len; ++i) {
      auto decoding_start = std::chrono::high_resolution_clock::now();
      this->UpdateLogitsOrProbOnCPUSync(this->ForwardTokens(first_sample_data, token_len + i + 1));
      TVMSynchronize(device_.device_type, device_.device_id, nullptr);
      auto decoding_end = std::chrono::high_resolution_clock::now();
      double decoding_ms = static_cast<double>((decoding_end - decoding_start).count()) / 1e6;
      decoding_ms_total += decoding_ms;
      LOG(INFO) << "[i: " << token_len + i + 1 << "] decoding-time=" << decoding_ms << "ms"
                << " tok/s: " << 1000.0 * (i + 1) / decoding_ms_total << ".";
    }
  }

  std::string RawGenerate(std::string prompt, int64_t generate_len) {
    CHECK_GE(generate_len, 0) << "The input generate is expected to be non-negative.";

    this->ResetKVCache();
    this->ResetRuntimeStats();

    std::vector<int32_t> tokens = tokenizer_->Encode(prompt);
    int64_t input_length = tokens.size();

    NDArray logits_on_device;
    // prefill
    {
      auto tstart = std::chrono::high_resolution_clock::now();
      logits_on_device = this->ForwardTokens(tokens, tokens.size());
      tokens.push_back(this->SampleTokenFromLogits(logits_on_device, temperature_, top_p_));
      auto tend = std::chrono::high_resolution_clock::now();

      this->prefill_total_time = static_cast<double>((tend - tstart).count()) / 1e9;
      this->prefill_total_tokens = input_length;
    }

    // decode
    {
      auto tstart = std::chrono::high_resolution_clock::now();
      for (int64_t len = 1; len < generate_len; ++len) {
        logits_on_device = this->ForwardTokens({tokens.back()}, tokens.size());
        tokens.push_back(this->SampleTokenFromLogits(logits_on_device, temperature_, top_p_));
      }
      auto tend = std::chrono::high_resolution_clock::now();

      this->decode_total_time = static_cast<double>((tend - tstart).count()) / 1e9;
      this->decode_total_tokens = generate_len;
    }

    std::string output = tokenizer_->Decode({tokens.begin() + input_length, tokens.end()});
    return output;
  }

 private:
  picojson::value SerializeConfigToJSONValue() const {
    picojson::object config;
    config["temperature"] = picojson::value(this->temperature_);
    config["repetition_penalty"] = picojson::value(this->repetition_penalty_);
    config["top_p"] = picojson::value(this->top_p_);
    config["mean_gen_len"] = picojson::value(this->mean_gen_len_);
    config["max_gen_len"] = picojson::value(this->max_gen_len_);
    config["shift_fill_factor"] = picojson::value(this->shift_fill_factor_);
    config["conv_config"] = this->conversation_.SerializeToJSON();
    return picojson::value(config);
  }
  /*!
   * \brief Sample output token from logits on device
   */
  int32_t SampleTokenFromLogits(NDArray logits_on_device, float temperature, float top_p) {
    if (repetition_penalty_ == 1.0f) {
      if (temperature_ < 1e-6f) {
        this->UpdateLogitsOrProbOnCPUSync(logits_on_device);
      } else {
        this->UpdateLogitsOrProbOnCPUSync(this->Softmax(logits_on_device, temperature_));
      }
    } else {
      this->UpdateLogitsOrProbOnCPUSync(logits_on_device);
      this->ApplyRepetitionPenaltyOnCPU();
      if (temperature_ >= 1e-6f) {
        this->ApplySoftmaxWithTemperatureOnCPU();
      }
    }
    auto tstart = std::chrono::high_resolution_clock::now();
    int next_token;
    if (temperature_ < 1e-6f) {
      next_token = this->SampleFromLogitsOnCPU();
    } else {
      next_token = this->SampleFromProbOnCPU();
    }
    auto tend = std::chrono::high_resolution_clock::now();
    this->sample_total_time += static_cast<double>((tend - tstart).count()) / 1e9;
    return next_token;
  }

  /*!
   * \brief Add a generated token and check for stop condition.
   * \param next_token The next token.
   */
  void ProcessNextToken(int32_t next_token) {
    ICHECK(!stop_triggered_) << "Cannot call process when it is stopped";

    stop_triggered_ =
        std::any_of(this->conversation_.stop_tokens.begin(), this->conversation_.stop_tokens.end(),
                    [next_token](int32_t token) { return token == next_token; });

    if (!stop_triggered_) {
      output_ids_.push_back(next_token);
      appeared_token_ids_.insert(next_token);
    }

    output_message_ = tokenizer_->Decode(output_ids_);

    if (!conversation_.stop_str.empty()) {
      size_t stop_pos = output_message_.rfind(conversation_.stop_str);
      if (stop_pos != std::string::npos) {
        stop_triggered_ = true;
        if (support_backtracking_kv_) {
          // back tracking, find the first set of token that is smaller
          // than the length
          size_t backoff = 0;
          for (; backoff < output_ids_.size(); ++backoff) {
            output_ids_.pop_back();
            output_message_ = tokenizer_->Decode(output_ids_);
            if (output_message_.length() <= stop_pos) break;
          }
          // resize kv to remove the context
          session_->CallPacked(fkvcache_array_popn_, kv_cache_, backoff);
          total_seq_len_ -= backoff;
        }
      }
    }

    if (static_cast<int64_t>(output_ids_.size()) >= max_gen_len_) {
      stop_triggered_ = true;
    } else if (total_seq_len_ >= max_window_size_) {
      stop_triggered_ = true;
    }
    if (stop_triggered_) {
      conversation_.FinishReply(output_message_);
    }
  }

  // run forward compute
  NDArray ForwardTokens(std::vector<int32_t> input_tokens, int64_t cur_pos) {
    NDArray ret_ndarray = NDArray::Empty({1, 1, vocab_size_}, DataType::Float(32), device_);
    if (input_tokens.size() > 1 && func_exists_[prefill_func_]) {
      NDArray input_data = this->GetInputTokenNDArray(input_tokens);
      DRef input_dref = CopyToWorker0(input_data);
      tvm::runtime::ShapeTuple cur_pos_shape = {cur_pos};
      DRef ret = session_->CallPacked(prefill_func_, input_dref, cur_pos_shape, kv_cache_, params_);
      ret = session_->CallPacked(tuple_getitem_func_, ret, 0);
      session_->CopyFromWorker0(ret_ndarray, ret);
      session_->SyncWorker(0);
    } else {
      // running decode function when prefill is not available
      for (int i = 0; i < input_tokens.size(); ++i) {
        NDArray input_data = this->GetInputTokenNDArray({input_tokens[i]});
        int64_t pos = cur_pos + i + 1 - input_tokens.size();
        DRef input_dref = CopyToWorker0(input_data);
        tvm::runtime::ShapeTuple pos_shape = {pos};
        DRef ret = session_->CallPacked(decode_func_, input_dref, pos_shape, kv_cache_, params_);
        ret = session_->CallPacked(tuple_getitem_func_, ret, 0);
        session_->CopyFromWorker0(ret_ndarray, ret);
        session_->SyncWorker(0);
      }
    }
    return ret_ndarray;
  }

  // run forward compute with embeddings
  NDArray ForwardEmbeddings(DRef embeddings, int64_t cur_pos) {
    NDArray ret_ndarray = NDArray::Empty({1, 1, vocab_size_}, DataType::Float(32), device_);
    tvm::runtime::ShapeTuple cur_pos_shape = {cur_pos};
    DRef ret = session_->CallPacked(prefill_with_embed_func_, embeddings, cur_pos_shape, kv_cache_,
                                    params_);
    ret = session_->CallPacked(tuple_getitem_func_, ret, 0);
    session_->CopyFromWorker0(ret_ndarray, ret);
    session_->SyncWorker(0);
    return ret_ndarray;
  }

  NDArray Softmax(NDArray input, float temperature) {
    NDArray ret_ndarray = NDArray::Empty(input.Shape(), DataType::Float(32), device_);
    NDArray temperature_arr = NDArray::Empty({}, DataType::Float(32), device_);
    temperature_arr.CopyFromBytes(&temperature, sizeof(float));
    DRef input_dref = CopyToWorker0(input);
    DRef temperature_dref = CopyToWorker0(temperature_arr);
    DRef ret = session_->CallPacked(softmax_func_, input_dref, temperature_dref);
    session_->CopyFromWorker0(ret_ndarray, ret);
    session_->SyncWorker(0);
    return ret_ndarray;
  }

  void ApplyRepetitionPenaltyOnCPU() {
    CHECK(logits_on_cpu_.defined()) << "Logits on CPU not defined!";
    CHECK(logits_on_cpu_.DataType() == DataType::Float(32)) << "Logits data type is not float32!";
    float* logits_raw_data = static_cast<float*>(logits_on_cpu_->data);
    for (const int32_t& token_id : this->appeared_token_ids_) {
      if (logits_raw_data[token_id] <= 0) {
        logits_raw_data[token_id] *= this->repetition_penalty_;
      } else {  // logits > 0
        logits_raw_data[token_id] /= this->repetition_penalty_;
      }
    }
  }

  void ApplySoftmaxWithTemperatureOnCPU() {
    CHECK(logits_on_cpu_.defined()) << "Logits on CPU not defined!";
    CHECK(logits_on_cpu_.DataType() == DataType::Float(32)) << "Logits data type is not float32!";
    int vocab_size = logits_on_cpu_->shape[logits_on_cpu_->ndim - 1];
    float* logits_raw_data = static_cast<float*>(logits_on_cpu_->data);
    float m = std::numeric_limits<float>::min();
    float inv_temp = 1.0f / this->temperature_;
    double d = 0.0f;
    for (int i = 0; i < vocab_size; ++i) {
      float x = logits_raw_data[i] * inv_temp;
      float m_prev = m;
      m = std::max(m, x);
      d = d * std::exp(m_prev - m) + std::exp(x - m);
    }
    for (int i = 0; i < vocab_size; ++i) {
      float x = logits_raw_data[i] * inv_temp;
      logits_raw_data[i] = std::exp(x - m) / d;
    }
  }

  void UpdateLogitsOrProbOnCPUSync(NDArray logits_or_prob) {
    if (!logits_on_cpu_.defined()) {
      logits_on_cpu_ = logits_or_prob.CopyTo(DLDevice{kDLCPU, 0});
    } else {
      ICHECK_EQ(logits_on_cpu_->shape[0], logits_or_prob->shape[0])
          << "Expect size of logits remain unchanged";
      logits_on_cpu_.CopyFrom(logits_or_prob);
    }
    TVMSynchronize(device_.device_type, device_.device_id, nullptr);
  }

  // Clear kv cache
  void ResetKVCache() { session_->CallPacked(reset_kv_cache_func_, kv_cache_); }

  void ProcessSystemPrompts() {
    this->PrefillStep(/*inp=*/"", /*append_conversation=*/false, /*decode_next_token=*/false);
  }

  // Utils
  static double GetRandomNumber() {
    static std::mt19937 gen(std::random_device{}());
    static std::uniform_real_distribution<> dis(0.0, 1.0);
    return dis(gen);
  }

  int32_t SampleFromLogitsOnCPU() {
    ICHECK(logits_on_cpu_.defined()) << "logits_on_cpu_ is not defined";
    ICHECK_EQ(logits_on_cpu_->ndim, 3) << "logits_on_cpu_ should be 3D";
    ICHECK_EQ(logits_on_cpu_->shape[0], 1) << "logits_on_cpu_ should be 1 batch";
    return fsample_topp_from_logits_(logits_on_cpu_, top_p_, temperature_, GetRandomNumber());
  }

  int32_t SampleFromProbOnCPU() {
    ICHECK(logits_on_cpu_.defined()) << "logits_on_cpu_ is not defined";
    ICHECK_EQ(logits_on_cpu_->ndim, 3) << "logits_on_cpu_ should be 3D";
    ICHECK_EQ(logits_on_cpu_->shape[0], 1) << "logits_on_cpu_ should be 1 batch";
    return fsample_topp_from_prob_(logits_on_cpu_, top_p_, GetRandomNumber());
  }

  //----------------------------
  // Statistics
  //----------------------------
  bool reset_stats_per_prefill_ = true;
  double embed_total_time = 0;
  double decode_total_time = 0;
  double sample_total_time = 0;
  double prefill_total_time = 0;
  int64_t decode_total_tokens = 0;
  int64_t prefill_total_tokens = 0;
  //----------------------------
  // Conversation
  //----------------------------
  // model name
  std::string model_name_;
  // conversation
  Conversation conversation_;
  // total sequence len,
  int64_t total_seq_len_{0};
  // max window size, mean generation length
  int64_t max_window_size_{768}, mean_gen_len_{128}, max_gen_len_{512};
  // vocab size
  int64_t vocab_size_;
  // shift window fill factor
  double shift_fill_factor_{0.3};
  // temperature
  double temperature_{0.8};
  // repetition penalty
  double repetition_penalty_{1.0};
  // top_p
  double top_p_{0.95};
  // output ids till now (refresh after encoding step)
  std::vector<int32_t> output_ids_;
  // appeared token ids till now (refresh after encoding step)
  std::unordered_set<int32_t> appeared_token_ids_;
  // output message till now (refresh after encoding step)
  std::string output_message_;
  // Whether encounter stop str
  bool stop_triggered_{false};
  // Whether we support rollback kv
  bool support_backtracking_kv_ = true;
  //----------------------------
  // Tokenizer
  //----------------------------
  // internal tokenizer
  std::unique_ptr<Tokenizer> tokenizer_;
  // bos token
  int32_t bos_token_id_{1};
  // eos token id
  int32_t eos_token_id_{2};
  //----------------------------
  // TVM related states
  //----------------------------
  // runtime device
  Device device_;
  // encoding function
  DRef prefill_func_{nullptr};
  // embedding function
  DRef embed_func_{nullptr};
  // encoding using embedding function
  DRef prefill_with_embed_func_{nullptr};
  // decoding function
  DRef decode_func_{nullptr};
  // encoding without cache
  DRef encoding_without_cache_func_{nullptr};
  // softmax
  DRef softmax_func_{nullptr};
  // reset kv cache
  DRef reset_kv_cache_func_{nullptr};
  // tuple get item
  DRef tuple_getitem_func_{nullptr};
  // sample top p from logits
  PackedFunc fsample_topp_from_logits_;
  // sample top p from prob
  PackedFunc fsample_topp_from_prob_;
  // pop n entries from kvcache
  DRef fkvcache_array_popn_{nullptr};
  // input token id
  NDArray input_token_ids_{nullptr};
  // local params
  DRef params_{nullptr};
  // KV cache
  DRef kv_cache_{nullptr};
  // Temp logits on cpu
  NDArray logits_on_cpu_{nullptr};
  // disco session
  tvm::runtime::Session session_{nullptr};
  // map of which function exists
  std::unordered_map<DRef, int, ObjectPtrHash, ObjectPtrEqual> func_exists_;
};

/*!
 * \brief A chat module implementation that exposes
 *  the functions as tvm::runtime::Module.
 *
 * We do it so that the module is accessible to any
 * language that tvm runtime can access.
 */
class LLMChatModule : public ModuleNode {
 public:
  // clear global memory manager
  static void ClearGlobalMemoryManager() {
    // Step 0. Clear the previously allocated memory.
    const PackedFunc* fclear_memory_manager =
        tvm::runtime::Registry::Get("vm.builtin.memory_manager.clear");
    ICHECK(fclear_memory_manager) << "Cannot find env function vm.builtin.memory_manager.clear";
    (*fclear_memory_manager)();
  }

  // overrides
  PackedFunc GetFunction(const String& name, const ObjectPtr<Object>& sptr_to_self) final {
    if (name == "reload") {
      return PackedFunc([this, sptr_to_self](TVMArgs args, TVMRetValue* rv) {
        chat_ = nullptr;
        ClearGlobalMemoryManager();
        chat_ = std::make_unique<LLMChat>(LLMChat(device_));
        ICHECK(2 <= args.size() && args.size() <= 3);
        if (args.size() == 2) {
          // args: executable, model_path
          chat_->Reload(args[0], args[1]);
        } else if (args.size() == 3) {
          // args: executable, model_path, app_config_json (used for overriding config)
          chat_->Reload(args[0], args[1], args[2]);
        }
      });
    } else if (name == "unload") {
      return PackedFunc([this, sptr_to_self](TVMArgs args, TVMRetValue* rv) {
        chat_ = nullptr;
        ClearGlobalMemoryManager();
      });
    } else if (name == "evaluate") {
      return PackedFunc([this, sptr_to_self](TVMArgs args, TVMRetValue* rv) {
        ICHECK_EQ(args.size(), 2);
        GetChat()->Evaluate(args[0], args[1]);
      });
    } else if (name == "raw_generate") {
      return PackedFunc([this, sptr_to_self](TVMArgs args, TVMRetValue* rv) {
        ICHECK_EQ(args.size(), 2);
        std::string s = GetChat()->RawGenerate(args[0], args[1]);
        *rv = s;
      });
    } else if (name == "prefill") {
      return PackedFunc([this, sptr_to_self](TVMArgs args, TVMRetValue* rv) {
        ICHECK(1 <= args.size() && args.size() <= 3);
        if (args.size() == 1) {
          // args: inp (with decode_next_token = true, place_in_prompt = kAll)
          GetChat()->PrefillStep(args[0]);
        } else if (args.size() == 2) {
          // args: inp, decode_next_token (with place_in_prompt = kAll)
          GetChat()->PrefillStep(args[0], true, args[1]);
        } else if (args.size() == 3) {
          // args: inp, decode_next_token, place_in_prompt
          PlaceInPrompt place_in_prompt = static_cast<PlaceInPrompt>(static_cast<int>(args[2]));
          GetChat()->PrefillStep(args[0], true, args[1], place_in_prompt);
        }
      });
    } else if (name == "embed") {
      return PackedFunc([this, sptr_to_self](TVMArgs args, TVMRetValue* rv) {
        ICHECK(1 <= args.size() && args.size() <= 2);
        if (args.size() == 1) {
          // args: inp (with place_in_prompt = kAll)
          *rv = GetChat()->EmbedStep(args[0]);
        } else if (args.size() == 2) {
          // args: inp, place_in_prompt
          PlaceInPrompt place_in_prompt = static_cast<PlaceInPrompt>(static_cast<int>(args[1]));
          *rv = GetChat()->EmbedStep(args[0], true, place_in_prompt);
        }
      });
    } else if (name == "prefill_with_embed") {
      return PackedFunc([this, sptr_to_self](TVMArgs args, TVMRetValue* rv) {
        ICHECK(1 <= args.size() && args.size() <= 2);
        if (args.size() == 1) {
          // args: embedding (with decode_next_token = true)
          GetChat()->PrefillWithEmbedStep(args[0]);
        } else if (args.size() == 2) {
          // args: embedding, decode_next_token
          GetChat()->PrefillWithEmbedStep(args[0], args[1]);
        }
      });
    } else if (name == "decode") {
      return PackedFunc(
          [this, sptr_to_self](TVMArgs args, TVMRetValue* rv) { GetChat()->DecodeStep(); });
    } else if (name == "reset_chat") {
      return PackedFunc([this, sptr_to_self](TVMArgs args, TVMRetValue* rv) {
        ICHECK_EQ(args.size(), 0);
        GetChat()->ResetChat();
      });
    } else if (name == "load_json_override") {
      return PackedFunc([this, sptr_to_self](TVMArgs args, TVMRetValue* rv) {
        ICHECK_EQ(args.size(), 2);
        std::string config_str = args[0];
        bool partial_update = args[1];
        GetChat()->LoadJSONOverride(config_str, partial_update);
      });
    } else if (name == "get_role0") {
      return PackedFunc([this, sptr_to_self](TVMArgs args, TVMRetValue* rv) {
        *rv = GetChat()->conversation_.roles[0];
      });
    } else if (name == "get_role1") {
      return PackedFunc([this, sptr_to_self](TVMArgs args, TVMRetValue* rv) {
        *rv = GetChat()->conversation_.roles[1];
      });
    } else if (name == "stopped") {
      return PackedFunc(
          [this, sptr_to_self](TVMArgs args, TVMRetValue* rv) { *rv = GetChat()->Stopped(); });
    } else if (name == "get_message") {
      return PackedFunc(
          [this, sptr_to_self](TVMArgs args, TVMRetValue* rv) { *rv = GetChat()->GetMessage(); });
    } else if (name == "runtime_stats_text") {
      return PackedFunc([this, sptr_to_self](TVMArgs args, TVMRetValue* rv) {
        *rv = GetChat()->RuntimeStatsText();
      });
    } else if (name == "reset_runtime_stats") {
      return PackedFunc(
          [this, sptr_to_self](TVMArgs args, TVMRetValue* rv) { GetChat()->ResetRuntimeStats(); });
    } else if (name == "get_config_json") {
      return PackedFunc([this, sptr_to_self](TVMArgs args, TVMRetValue* rv) {
        *rv = GetChat()->GetConfigJSON();
      });
    } else if (name == "process_system_prompts") {
      return PackedFunc([this, sptr_to_self](TVMArgs args, TVMRetValue* rv) {
        GetChat()->ProcessSystemPrompts();
      });
    } else {
      return PackedFunc(nullptr);
    }
  }

  void Init(DLDevice device) { device_ = device; }

  LLMChat* GetChat() {
    ICHECK(chat_ != nullptr) << "Chat is not initialized via reload";
    return chat_.get();
  }

  const char* type_key() const final { return "mlc.llm_chat"; }

 private:
  std::unique_ptr<LLMChat> chat_ = nullptr;
  DLDevice device_;
};

std::vector<std::string> CountUTF8(const std::string& s) {
  // assume that the string is always valid utf8
  std::vector<std::string> ret;
  for (size_t pos = 0; pos < s.size();) {
    if ((s[pos] & 0x80) == 0x00) {
      ret.push_back(s.substr(pos, 1));
      pos += 1;
    } else if (pos + 1 < s.size() && (s[pos] & 0xE0) == 0xC0 && (s[pos + 1] & 0xC0) == 0x80) {
      ret.push_back(s.substr(pos, 2));
      pos += 2;
    } else if (pos + 1 < s.size() && (s[pos] & 0xF0) == 0xE0 && (s[pos + 1] & 0xC0) == 0x80 &&
               (s[pos + 2] & 0xC0) == 0x80) {
      ret.push_back(s.substr(pos, 3));
      pos += 3;
    } else if (pos + 2 < s.size() && (s[pos] & 0xF8) == 0xF0 && (s[pos + 1] & 0xC0) == 0x80 &&
               (s[pos + 2] & 0xC0) == 0x80 && (s[pos + 3] & 0xC0) == 0x80) {
      ret.push_back(s.substr(pos, 4));
      pos += 4;
    } else {
      LOG(FATAL) << "Invalid UTF8 string";
    }
  }
  return std::move(ret);
}

/*!
 * \brief Get the diff of new message and current message (the delta message).
 * \param curr_message The current message.
 * \param new_message The new message
 * \return The delta message.
 * \note The main complication here is that new_msg can be different from previous message, so we
 need to find the diff, delete previous messages that are different, then print it out.
 This logic is only needed for simple stdout.

 For UI apps that can directly update output text we can simply do last_reply.text =
 chat->GetMessage();
 */
std::string GetDeltaMessage(std::string curr_message, std::string new_message) {
  std::vector<std::string> cur_utf8_chars = CountUTF8(curr_message);
  std::vector<std::string> new_utf8_chars = CountUTF8(new_message);
  // Step 1. Find the index of the first UTF8 char that differs
  size_t pos = std::mismatch(cur_utf8_chars.begin(), cur_utf8_chars.end(), new_utf8_chars.begin(),
                             new_utf8_chars.end())
                   .first -
               cur_utf8_chars.begin();
  // Step 2. Delete the previous message since `pos`
  std::string print = "";
  for (size_t j = pos; j < cur_utf8_chars.size(); ++j) {
    print += "\b \b";
  }
  // Step 3. Print the new message since `pos`
  for (size_t j = pos; j < new_utf8_chars.size(); ++j) {
    print += new_utf8_chars[j];
  }
  return print;
}

// register as a system function that can be queried
TVM_REGISTER_GLOBAL("mlc.get_delta_message").set_body_typed(GetDeltaMessage);

tvm::runtime::Module CreateChatModule(DLDevice device) {
  ObjectPtr<LLMChatModule> n = make_object<LLMChatModule>();
  n->Init(device);
  return Module(n);
}

// register as a system function that can be queried
TVM_REGISTER_GLOBAL("mlc.llm_chat_create").set_body_typed([](int device_type, int device_id) {
  return CreateChatModule(DLDevice{static_cast<DLDeviceType>(device_type), device_id});
});

}  // namespace llm
}  // namespace mlc

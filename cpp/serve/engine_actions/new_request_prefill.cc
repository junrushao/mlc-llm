/*!
 *  Copyright (c) 2023 by Contributors
 * \file serve/engine_actions/new_request_prefill.cc
 */

#include "../config.h"
#include "../model.h"
#include "../sampler.h"
#include "action.h"

namespace mlc {
namespace llm {
namespace serve {

/*!
 * \brief The action that prefills requests in the `waiting_queue` of
 * the engine state.
 */
class NewRequestPrefillActionObj : public EngineActionObj {
 public:
  explicit NewRequestPrefillActionObj(Array<Model> models, Sampler sampler,
                                      KVCacheConfig kv_cache_config, int max_single_sequence_length)
      : models_(std::move(models)),
        sampler_(std::move(sampler)),
        kv_cache_config_(std::move(kv_cache_config)),
        max_single_sequence_length_(max_single_sequence_length) {}

  Array<Request> Step(EngineState estate) final {
    // - Find the requests in `waiting_queue` that can prefill in this step.
    auto [requests, rstates, prefill_lengths] = GetRequestsToPrefill(estate);
    ICHECK_EQ(requests.size(), rstates.size());
    ICHECK_EQ(requests.size(), prefill_lengths.size());
    if (requests.empty()) {
      return {};
    }

    int num_requests = requests.size();
    auto tstart = std::chrono::high_resolution_clock::now();

    // - Move requests from waiting queue to running queue.
    //   And assign internal ID for the requests.
    std::vector<int> request_ids;
    request_ids.reserve(num_requests);
    for (int i = 0; i < num_requests; ++i) {
      int req_id = estate->running_queue.size();
      auto it = std::find(estate->waiting_queue.begin(), estate->waiting_queue.end(), requests[i]);
      ICHECK(it != estate->waiting_queue.end());

      // - Move request from waiting queue to running queue.
      estate->waiting_queue.erase(it);
      estate->running_queue.push_back(requests[i]);
      // - Assign internal request id for the requests.
      AssignInternalIDForRequest(rstates[i], requests[i], req_id);
      request_ids.push_back(req_id);
    }

    // - Get embedding and run prefill for each model.
    NDArray logits_for_sample{nullptr};
    for (int model_id = 0; model_id < static_cast<int>(models_.size()); ++model_id) {
      Array<NDArray> embeddings;
      embeddings.reserve(num_requests);
      for (int i = 0; i < num_requests; ++i) {
        RequestModelState mstate = rstates[i]->mstates[model_id];
        ICHECK_EQ(mstate->GetInputLength(), prefill_lengths[i]);
        ICHECK(mstate->draft_output_tokens.empty());
        ICHECK(mstate->draft_output_token_prob.empty());
        ICHECK(mstate->draft_output_prob_dist.empty());
        ICHECK(!mstate->inputs.empty());
        for (int i = 0; i < static_cast<int>(mstate->inputs.size()); ++i) {
          embeddings.push_back(mstate->inputs[i]->GetEmbedding(models_[model_id]));
        }
        // Clean up `inputs` after prefill
        mstate->inputs.clear();
      }

      NDArray logits = models_[model_id]->BatchPrefill(embeddings, request_ids, prefill_lengths);
      ICHECK_EQ(logits->ndim, 3);
      ICHECK_EQ(logits->shape[0], 1);
      ICHECK_EQ(logits->shape[1], num_requests);

      if (model_id == 0) {
        // We only need to sample for model 0 in prefill.
        logits_for_sample = logits;
      }
    }

    // - Sample tokens.
    ICHECK(logits_for_sample.defined());
    logits_for_sample = logits_for_sample.CreateView({num_requests, 1, logits_for_sample->shape[2]},
                                                     logits_for_sample->dtype);
    Array<RequestModelState> mstates_for_sample =
        rstates.Map([](RequestState rstate) { return rstate->mstates[0]; });
    std::vector<int32_t> next_tokens = sampler_->SampleTokens(
        logits_for_sample, models_[0], mstates_for_sample,
        requests.Map([](Request request) { return request->generation_cfg; }));
    ICHECK_EQ(next_tokens.size(), num_requests);

    // - Update the committed tokens of states.
    // - If a request is first-time prefilled, set the prefill finish time.
    // - Accumulate the sequence length in engine statistics.
    int sum_prefill_lengths = 0;
    auto tnow = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_requests; ++i) {
      mstates_for_sample[i]->committed_tokens.push_back(next_tokens[i]);
      if (mstates_for_sample[i]->committed_tokens.size() == 1) {
        estate->GetRequestState(requests[i])->tprefill_finish = tnow;
      }
      sum_prefill_lengths += prefill_lengths[i];
    }
    estate->stats.current_total_seq_len += sum_prefill_lengths;

    auto tend = std::chrono::high_resolution_clock::now();
    estate->stats.engine_total_prefill_time += static_cast<double>((tend - tstart).count()) / 1e9;

    return requests;
  }

 private:
  /*!
   * \brief Find one or multiple requests to run prefill.
   * \param estate The engine state.
   * \return The requests to prefill, together with their respective
   * state and input length.
   */
  std::tuple<Array<Request>, Array<RequestState>, std::vector<int>> GetRequestsToPrefill(
      EngineState estate) {
    if (estate->waiting_queue.empty()) {
      // No request to prefill.
      return {{}, {}, {}};
    }

    // - Try to prefill pending requests.
    std::vector<Request> prefill_requests;
    std::vector<RequestState> rstates;
    std::vector<int> prefill_lengths;
    int total_input_length = 0;
    int total_required_pages = 0;
    int num_available_pages = models_[0]->GetNumAvailablePages();

    for (int i = 1; i <= static_cast<int>(estate->waiting_queue.size()); ++i) {
      Request request = estate->waiting_queue[i - 1];
      RequestState rstate = estate->GetRequestState(request);
      int input_length = rstate->mstates[0]->GetInputLength();
      int num_require_pages =
          (input_length + kv_cache_config_->page_size - 1) / kv_cache_config_->page_size;
      total_input_length += input_length;
      total_required_pages += num_require_pages;
      if (CanPrefill(estate, i, total_input_length, total_required_pages, num_available_pages)) {
        prefill_requests.push_back(request);
        rstates.push_back(rstate);
        prefill_lengths.push_back(input_length);
      } else {
        total_input_length -= input_length;
        total_required_pages -= num_require_pages;
        break;
      }
    }

    return {prefill_requests, rstates, prefill_lengths};
  }

  /*! \brief Check if the input requests can be prefilled under conditions. */
  bool CanPrefill(EngineState estate, int num_prefill_req, int total_input_length,
                  int num_required_pages, int num_available_pages) {
    int num_running_requests = estate->running_queue.size();
    ICHECK_LE(num_running_requests, kv_cache_config_->max_num_sequence);

    // No exceeding of the maximum allowed requests that can
    // run simultaneously.
    if (num_running_requests + num_prefill_req > kv_cache_config_->max_num_sequence) {
      return false;
    }

    // NOTE: The conditions are heuristic and can be revised.
    // Cond 1: total input length <= max allowed single sequence length.
    // Cond 2: at least one decode can be performed after prefill.
    // Cond 3: number of total tokens after 8 times of decode does not
    // exceed the limit, where 8 is a watermark number can
    // be configured and adjusted in the future.
    int new_batch_size = num_running_requests + num_prefill_req;
    return total_input_length <= max_single_sequence_length_ &&
           num_required_pages + new_batch_size <= num_available_pages &&
           estate->stats.current_total_seq_len + total_input_length + 8 * new_batch_size <=
               kv_cache_config_->max_total_sequence_length;
  }

  /*! \brief Assign the given internal id for the given request. */
  void AssignInternalIDForRequest(RequestState rstate, Request request, int req_id) {
    // Set internal id in the request state.
    for (RequestModelState mstate : rstate->mstates) {
      mstate->request_id = req_id;
    }
    // Add a new sequence to each model.
    for (int i = 0; i < static_cast<int>(models_.size()); ++i) {
      int seq_id_in_model = models_[i]->AddNewSequence();
      ICHECK_EQ(seq_id_in_model, req_id);
    }
  }

  /*! \brief The models to run prefill in. */
  Array<Model> models_;
  /*! \brief The sampler to sample new tokens. */
  Sampler sampler_;
  /*! \brief The KV cache config to help decide prefill is doable. */
  KVCacheConfig kv_cache_config_;
  /*! \brief The max single sequence length to help decide if prefill is doable. */
  int max_single_sequence_length_;
};

EngineAction EngineAction::NewRequestPrefill(Array<Model> models, Sampler sampler,
                                             KVCacheConfig kv_cache_config,
                                             int max_single_sequence_length) {
  return EngineAction(make_object<NewRequestPrefillActionObj>(std::move(models), std::move(sampler),
                                                              std::move(kv_cache_config),
                                                              max_single_sequence_length));
}

}  // namespace serve
}  // namespace llm
}  // namespace mlc

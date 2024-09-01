"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import logging
import json
import os

import lavis.common.dist_utils as dist_utils
from lavis.common.registry import registry
from lavis.common.vqa_tools.vqa import VQA
from lavis.common.vqa_tools.vqa_eval import VQAEval
from lavis.tasks.base_task import BaseTask
import numpy as np
import copy, random

from lavis.common.dist_utils import get_rank, get_world_size, is_main_process, is_dist_avail_and_initialized
import torch


@registry.register_task("vqa")
class VQATask(BaseTask):
    def __init__(
        self,
        num_beams,
        max_len,
        min_len,
        evaluate,
        num_ans_candidates,
        inference_method="rank",
        prompt="",
    ):
        super().__init__()

        self.num_beams = num_beams
        self.max_len = max_len
        self.min_len = min_len

        self.evaluate = evaluate
        self.inference_method = inference_method
        self.num_ans_candidates = num_ans_candidates
        self.prompt = prompt

        self.answer_list = None

        self.ques_files = dict()
        self.anno_files = dict()

    @classmethod
    def setup_task(cls, cfg):
        run_cfg = cfg.run_cfg

        num_beams = run_cfg.get("num_beams", 3)
        max_len = run_cfg.get("max_len", 10)
        min_len = run_cfg.get("min_len", 1)

        evaluate = run_cfg.get("evaluate", False)

        inference_method = run_cfg.get("inference_method", "rank")
        num_ans_candidates = run_cfg.get("num_ans_candidates", 128)
        prompt = run_cfg.get("prompt", "")

        return cls(
            num_beams=num_beams,
            max_len=max_len,
            min_len=min_len,
            evaluate=evaluate,
            num_ans_candidates=num_ans_candidates,
            inference_method=inference_method,
            prompt=prompt,
        )

    def build_datasets(self, cfg):
        datasets = super().build_datasets(cfg)

        # get question file, annotation file and anwser list in COCO format
        for dataset in datasets.values():
            for split in dataset:
                if hasattr(dataset[split], "coco_fmt_qust_file") and dataset[split].coco_fmt_qust_file is not None:
                    self.ques_files[split] = dataset[split].coco_fmt_qust_file
                    self.anno_files[split] = dataset[split].coco_fmt_anno_file

                try:
                    self.answer_list = dataset[split].answer_list
                except AttributeError:
                    # if answer_list is not provided, then set it to None
                    pass

        if len(self.ques_files) > 0:
            assert len(self.ques_files) == len(self.anno_files), "Only support one split for evaluation."

        return datasets

    def valid_step(self, model, samples):

        answers = model.predict_answers(
            samples=samples,
            answer_list=self.answer_list,
            inference_method=self.inference_method,
            num_beams=self.num_beams,
            max_len=self.max_len,
            min_len=self.min_len,
            num_ans_candidates=self.num_ans_candidates,
            prompt=self.prompt,
        )
        pred_qa_pairs = []

        gt_answers = samples["answers"]
        questions = samples["questions"]

        for answer, gt_answer, ques in zip(answers, gt_answers, questions):
            pred_qa_pairs.append({"question": ques, "answer": answer, "gt_answer": gt_answer})

        return pred_qa_pairs

    def after_evaluation(self, val_result, split_name, epoch, model):
        result_file = self.save_result(
            val_result,
            result_dir=registry.get_path("result_dir"),
            filename=f"{split_name}_%d_vqa_result" % epoch,
            # remove_duplicate="question_id",
        )

        metrics = self._report_metrics(result_file=result_file, split=split_name)

        return metrics

    @dist_utils.main_process
    def _report_metrics(self, result_file, split):
        """
        Use official VQA evaluation script to report metrics.
        """
        metrics = {}

        if split in self.ques_files and split in self.anno_files:
            vqa = VQA(self.anno_files[split], self.ques_files[split])
            vqa_result = vqa.loadRes(resFile=result_file, quesFile=self.ques_files[split])

            # create vqaEval object by taking vqa and vqaRes
            # n is precision of accuracy (number of places after decimal), default is 2
            vqa_scorer = VQAEval(vqa, vqa_result, n=2)
            logging.info("Start VQA evaluation.")
            vqa_scorer.evaluate()

            # print accuracies
            overall_acc = vqa_scorer.accuracy["overall"]
            metrics["agg_metrics"] = overall_acc

            logging.info("Overall Accuracy is: %.02f\n" % overall_acc)
            logging.info("Per Answer Type Accuracy is the following:")

            for ans_type in vqa_scorer.accuracy["perAnswerType"]:
                logging.info("%s : %.02f" % (ans_type, vqa_scorer.accuracy["perAnswerType"][ans_type]))
                metrics[ans_type] = vqa_scorer.accuracy["perAnswerType"][ans_type]

            with open(os.path.join(registry.get_path("output_dir"), "evaluate.txt"), "a") as f:
                f.write(json.dumps(metrics) + "\n")

        return metrics


@registry.register_task("occ")
class OccTask(VQATask):
    def valid_step(self, model, samples):

        answers = model.predict_answers(
            samples=samples,
            answer_list=self.answer_list,
            inference_method=self.inference_method,
            num_beams=self.num_beams,
            max_len=self.max_len,
            min_len=self.min_len,
            num_ans_candidates=self.num_ans_candidates,
            prompt=self.prompt,
        )
        pred_qa_pairs = []

        # gt_answers = samples["answers"]
        # questions = samples["questions"]

        # for answer, gt_answer, ques in zip(answers, gt_answers, questions):
        #     pred_qa_pairs.append({"question": ques, "answer": answer, "gt_answer": gt_answer})

        for occ, ego_pose, ego_label, ego_L2 in zip(answers['occ'], answers['ego_pose'], answers['ego_label'], answers['ego_L2']):
            pred_qa_pairs.append({"occ": occ, "ego_pose": ego_pose, "ego_label": ego_label, "ego_L2": ego_L2})

        return pred_qa_pairs

    def after_evaluation(self, val_result, split_name, epoch, model):
        local_rank = get_rank()
        results = val_result

        CalMeanIou_sem = model.miou
        CalMeanIou_vox = model.iou

        # miou, _ = CalMeanIou_sem._after_epoch(local_rank)
        # iou, _ = CalMeanIou_vox._after_epoch(local_rank)

        miou, _ = CalMeanIou_sem._after_epoch_single_gpu()
        iou, _ = CalMeanIou_vox._after_epoch_single_gpu()

        # if is_main_process():
        #     print(f"miou 0s {miou[6]:.2f} 1s {miou[8]:.2f} 2s {miou[10]:.2f} 3s {miou[12]:.2f}")
        #     print(f"iou 0s {iou[6]:.2f} 1s {iou[8]:.2f} 2s {iou[10]:.2f} 3s {iou[12]:.2f}")

        # ego = torch.from_numpy(results.predictions[-1]["ego_L2"]).mean(dim=0)

        for cur in results["ego_L2"]:
            print()
            
        ego = torch.from_numpy(results["ego_L2"]).mean(dim=0)

        

        # if is_main_process():
        #     print(f"ego 0s {ego[6]:.2f} 1s {ego[8]:.2f} 2s {ego[10]:.2f} 3s {ego[12]:.2f}")
        
        # if not training_args.only_calc_metrics and local_rank == 0:
        if is_main_process():
            if isinstance(results.predictions[-1], dict):
                occ = results.predictions[-1]["occ"]
            elif isinstance(results.predictions, tuple):
                occ = results.predictions[-1]
            else:
                occ = results.predictions
            
            occ_label = results.label_ids
            print(occ.shape, occ_label.shape)
            
            dataset_name = 'valset'
            self.save_occ_results(registry.get_path("result_dir"), occ, occ_label, dataset_name)

        metrics = self._report_metrics()

        return metrics

    @dist_utils.main_process
    def _report_metrics(self, result_file=None, split=None):
        """
        Use official VQA evaluation script to report metrics.
        """
        metrics = {}

        # if split in self.ques_files and split in self.anno_files:
        #     vqa = VQA(self.anno_files[split], self.ques_files[split])
        #     vqa_result = vqa.loadRes(resFile=result_file, quesFile=self.ques_files[split])

        #     # create vqaEval object by taking vqa and vqaRes
        #     # n is precision of accuracy (number of places after decimal), default is 2
        #     vqa_scorer = VQAEval(vqa, vqa_result, n=2)
        #     logging.info("Start VQA evaluation.")
        #     vqa_scorer.evaluate()

        #     # print accuracies
        #     overall_acc = vqa_scorer.accuracy["overall"]
        #     metrics["agg_metrics"] = overall_acc

        #     logging.info("Overall Accuracy is: %.02f\n" % overall_acc)
        #     logging.info("Per Answer Type Accuracy is the following:")

        #     for ans_type in vqa_scorer.accuracy["perAnswerType"]:
        #         logging.info("%s : %.02f" % (ans_type, vqa_scorer.accuracy["perAnswerType"][ans_type]))
        #         metrics[ans_type] = vqa_scorer.accuracy["perAnswerType"][ans_type]

        #     with open(os.path.join(registry.get_path("output_dir"), "evaluate.txt"), "a") as f:
        #         f.write(json.dumps(metrics) + "\n")

        return metrics

# @registry.register_task("gqa")
# GQATask = VQATask
# @registry.register_task("aok_vqa")
# AOKVQATask = VQATask
@registry.register_task("gqa")
class GQATask(VQATask):
    pass
    # def valid_step(self, model, samples):
    #    answers = model.predict_answers(
    #        samples=samples,
    #        answer_list=self.answer_list,
    #        inference_method=self.inference_method,
    #        num_beams=self.num_beams,
    #        max_len=self.max_len,
    #        min_len=self.min_len,
    #        num_ans_candidates=self.num_ans_candidates,
    #        prompt=self.prompt,
    #    )
    #    pred_qa_pairs = []

    #    question_id = samples["question_id"]
    #    gt_answers = samples["answer"]

    #    for answer, ques_id, gt_answer in zip(answers, question_id, gt_answers):
    #        ques_id = int(ques_id.item())
    #        pred_qa_pairs.append({"question_id": ques_id, "pred_ans": answer, "gt_ans": gt_answer})

    #    return pred_qa_pairs

    # @dist_utils.main_process
    # def _report_metrics(self, result_file, split):
    #    """
    #    TODO: add other evaluation metrics for GQA
    #    """

    #   results = json.load(open(result_file, "r"))
    #   acc = []
    #   vqa_tool = VQAEval()

    #  for res in results:
    #      if res["gt_ans"] is None:
    # prepare test results for leaderboard evaluation
    #          self._save_result_leaderboard(results)
    #          return

    #     gt_ans = res["gt_ans"]
    #     pred = res["pred_ans"]

    #     if self.inference_method == "generate":
    #         pred = vqa_tool.processPunctuation(pred)
    #         pred = vqa_tool.processDigitArticle(pred)

    #    vqa_acc = 1 if pred == gt_ans else 0

    #    acc.append(vqa_acc)

    # accuracy = sum(acc) / len(acc) * 100
    # metrics = {"agg_metrics": accuracy, "acc": accuracy}

    # with open(
    #    os.path.join(registry.get_path("output_dir"), "evaluate.txt"), "a"
    # ) as f:
    #    f.write(json.dumps(metrics) + "\n")

    # logging.info(metrics)

    # return metrics


@registry.register_task("3d_vqa")
class ThreeDVQATask(VQATask):
    pass


#    def valid_step(self, model, samples):
#        answers = model.predict_answers(
#            samples=samples,
#            answer_list=self.answer_list,
#            inference_method=self.inference_method,
#            num_beams=self.num_beams,
#            max_len=self.max_len,
#            min_len=self.min_len,
#            num_ans_candidates=self.num_ans_candidates,
#        )

#        pred_qa_pairs = []

#        question_id = samples["question_id"]
#        gt_answers = samples["direct_answers"]

#        for pred_answer, ques_id, gt_answer in zip(answers, question_id, gt_answers):
#            pred_qa_pairs.append(
#                {"question_id": ques_id, "pred_ans": pred_answer, "gt_ans": gt_answer}
#            )

#        return pred_qa_pairs

#    @dist_utils.main_process
#    def _report_metrics(self, result_file, split):
#        """
#        Implementing accuracy computation for AOKVQA, see
#        https://github.com/allenai/aokvqa/blob/main/evaluation/eval_predictions.py#L45 for details.
#        """
# TODO add evaluation for multi-choice

#        results = json.load(open(result_file, "r"))
#        acc = []

#        for res in results:
#            if res["gt_ans"] is None:
# prepare test results for leaderboard evaluation
#                self._save_result_leaderboard(results)
#                return

#            pred = res["pred_ans"]
#            gt_ans = res["gt_ans"]

#            num_match = sum([pred == gt for gt in gt_ans])
#            vqa_acc = min(1.0, num_match / 3.0)

#            acc.append(vqa_acc)
#
#        accuracy = sum(acc) / len(acc) * 100
#        metrics = {"agg_metrics": accuracy, "acc": accuracy}

#        with open(
#            os.path.join(registry.get_path("output_dir"), "evaluate.txt"), "a"
#        ) as f:
#            f.write(json.dumps(metrics) + "\n")

#        logging.info(metrics)

#        return metrics

#    @dist_utils.main_process
#    def _save_result_leaderboard(self, results):
#        """
#        Saving the results in the format required for leaderboard evaluation.

#        [TODO] add support for multi-choice.
#        """
#        result_leaderboard = dict()
#        for res in results:
#            result_leaderboard[res["question_id"]] = {
#                "direct_answer": res["pred_ans"],
#                "multiple_choice": "",
#            }

#        result_file = registry.get_path("result_dir") + "_leaderboard.json"

#        with open(result_file, "w") as f:
#            json.dump(result_leaderboard, f)

#        logging.info(f"Saved results for leaderboard evaluation at {result_file}")

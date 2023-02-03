import torch
import numpy as np
from utils import log_string


class Evaluation:
    """
    Handles evaluation on a given POI dataset and loader.

    The two metrics are MAP and recall@n. Our model predicts sequence of
    next locations determined by the sequence_length at one pass. During evaluation we
    treat each entry of the sequence as single prediction. One such prediction
    is the ranked list of all available locations and we can compute the two metrics.

    As a single prediction is of the size of all available locations,
    evaluation takes its time to compute. The code here is optimized.

    Using the --report_user argument one can access the statistics per user.
    """

    def __init__(self, dataset, dataloader, user_count, h0_strategy, trainer, setting, log):
        self.dataset = dataset
        self.dataloader = dataloader
        self.user_count = user_count
        self.h0_strategy = h0_strategy
        self.trainer = trainer
        self.setting = setting
        self._log = log

    def evaluate(self, niche, hot):
        self.dataset.reset()
        h = self.h0_strategy.on_init(self.setting.batch_size, self.setting.device)


        with torch.no_grad():
            iter_cnt = 0
            recall1 = 0
            recall5 = 0
            recall10 = 0
            average_precision = 0.

            iter_cnt_niche = 0
            iter_cnt_hot = 0
            recall1_niche = 0
            recall5_niche = 0
            recall10_niche = 0
            recall1_hot = 0
            recall5_hot = 0
            recall10_hot = 0

            u_iter_cnt = np.zeros(self.user_count)
            u_recall1 = np.zeros(self.user_count)
            u_recall5 = np.zeros(self.user_count)
            u_recall10 = np.zeros(self.user_count)
            u_average_precision = np.zeros(self.user_count)
            reset_count = torch.zeros(self.user_count)

            u_iter_cnt_niche = np.zeros(self.user_count)
            u_iter_cnt_hot = np.zeros(self.user_count)
            u_recall1_niche = np.zeros(self.user_count)
            u_recall1_hot = np.zeros(self.user_count)
            u_recall5_niche = np.zeros(self.user_count)
            u_recall5_hot = np.zeros(self.user_count)
            u_recall10_niche = np.zeros(self.user_count)
            u_recall10_hot = np.zeros(self.user_count)


            for i, (x, t, t_slot, s, r, y, y_t, y_t_slot, y_s, y_r, reset_h, active_users) in enumerate(self.dataloader):
                active_users = active_users.squeeze()
                for j, reset in enumerate(reset_h):
                    if reset:
                        if self.setting.is_lstm:
                            hc = self.h0_strategy.on_reset_test(active_users[j], self.setting.device)
                            h[0][0, j] = hc[0]
                            h[1][0, j] = hc[1]
                        else:
                            h[0, j] = self.h0_strategy.on_reset_test(active_users[j], self.setting.device)
                        reset_count[active_users[j]] += 1

                # squeeze for reasons of "loader-batch-size-is-1"
                x = x.squeeze().to(self.setting.device)
                t = t.squeeze().to(self.setting.device)
                t_slot = t_slot.squeeze().to(self.setting.device)
                s = s.squeeze().to(self.setting.device)
                r = r.squeeze().to(self.setting.device)

                y = y.squeeze()
                y_t = y_t.squeeze().to(self.setting.device)
                y_t_slot = y_t_slot.squeeze().to(self.setting.device)
                y_s = y_s.squeeze().to(self.setting.device)
                y_r = y_r.squeeze().to(self.setting.device)
                active_users = active_users.to(self.setting.device)

                # evaluate:
                out, h = self.trainer.evaluate(x, t, t_slot, s, r, y_t, y_t_slot, y_s, y_r, h, active_users)

                for j in range(self.setting.batch_size):
                    # o contains a per user list of votes for all locations for each sequence entry
                    o = out[j]

                    # partition elements
                    o_n = o.cpu().detach().numpy()
                    ind = np.argpartition(o_n, -10, axis=1)[:, -10:]  # top 10 elements

                    y_j = y[:, j]

                    for k in range(len(y_j)):
                        if reset_count[active_users[j]] > 1:
                            continue  # skip already evaluated users.

                        # resort indices for k:
                        ind_k = ind[k]
                        r = ind_k[np.argsort(-o_n[k, ind_k], axis=0)]  # sort top 10 elements descending

                        r = torch.tensor(r)
                        t = y_j[k]

                        # compute MAP:
                        r_kj = o_n[k, :]
                        t_val = r_kj[t]
                        upper = np.where(r_kj > t_val)[0]
                        precision = 1. / (1 + len(upper))

                        # store
                        u_iter_cnt[active_users[j]] += 1
                        u_recall1[active_users[j]] += t in r[:1]
                        u_recall5[active_users[j]] += t in r[:5]
                        u_recall10[active_users[j]] += t in r[:10]
                        u_average_precision[active_users[j]] += precision


                        if int(t) in niche:
                            u_iter_cnt_niche[active_users[j]] += 1
                            u_recall1_niche[active_users[j]] += t in r[:1]
                            u_recall5_niche[active_users[j]] += t in r[:5]
                            u_recall10_niche[active_users[j]] += t in r[:10]
                        else:
                            u_iter_cnt_hot[active_users[j]] += 1
                            u_recall1_hot[active_users[j]] += t in r[:1]
                            u_recall5_hot[active_users[j]] += t in r[:5]
                            u_recall10_hot[active_users[j]] += t in r[:10]



            formatter = "{0:.8f}"
            for j in range(self.user_count):
                iter_cnt += u_iter_cnt[j]
                recall1 += u_recall1[j]
                recall5 += u_recall5[j]
                recall10 += u_recall10[j]
                average_precision += u_average_precision[j]

                iter_cnt_niche += u_iter_cnt_niche[j]
                iter_cnt_hot += u_iter_cnt_hot[j]
                recall1_niche += u_recall1_niche[j]
                recall5_niche += u_recall5_niche[j]
                recall10_niche += u_recall10_niche[j]
                recall1_hot += u_recall1_hot[j]
                recall5_hot += u_recall5_hot[j]
                recall10_hot += u_recall10_hot[j]

                if self.setting.report_user > 0 and (j + 1) % self.setting.report_user == 0:
                    print('Report user', j, 'preds:', u_iter_cnt[j], 'recall@1',
                          formatter.format(u_recall1[j] / u_iter_cnt[j]), 'MAP',
                          formatter.format(u_average_precision[j] / u_iter_cnt[j]), sep='\t')

            # print('recall@1:', formatter.format(recall1 / iter_cnt))
            # print('recall@5:', formatter.format(recall5 / iter_cnt))
            # print('recall@10:', formatter.format(recall10 / iter_cnt))
            # print('MAP', formatter.format(average_precision / iter_cnt))
            # print('predictions:', iter_cnt)

            log_string(self._log, 'recall@1: ' + formatter.format(recall1 / iter_cnt))
            log_string(self._log, 'recall@5: ' + formatter.format(recall5 / iter_cnt))
            log_string(self._log, 'recall@10: ' + formatter.format(recall10 / iter_cnt))
            log_string(self._log, 'MAP: ' + formatter.format(average_precision / iter_cnt))
            log_string(self._log, 'recall@1 niche: ' + formatter.format(recall1_niche / iter_cnt_niche))
            log_string(self._log, 'recall@5 niche: ' + formatter.format(recall5_niche / iter_cnt_niche))
            log_string(self._log, 'recall@10 niche: ' + formatter.format(recall10_niche / iter_cnt_niche))
            log_string(self._log, 'recall@1 hot: ' + formatter.format(recall1_hot / iter_cnt_hot))
            log_string(self._log, 'recall@5 hot: ' + formatter.format(recall5_hot / iter_cnt_hot))
            log_string(self._log, 'recall@10 hot: ' + formatter.format(recall10_hot / iter_cnt_hot))
            print('predictions:', iter_cnt)

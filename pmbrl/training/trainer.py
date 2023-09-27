# pylint: disable=not-callable
# pylint: disable=no-member

import torch


class Trainer(object):
    def __init__(
        self,
        ensemble,
        reward_model,
        buffer,
        n_train_epochs,
        batch_size,
        learning_rate,
        epsilon,
        grad_clip_norm,
        logger=None,
    ):
        self.ensemble = ensemble
        self.reward_model = reward_model
        self.buffer = buffer
        self.n_train_epochs = n_train_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.grad_clip_norm = grad_clip_norm
        self.logger = logger

        self.params = list(ensemble.parameters()) + list(reward_model.parameters())
        self.optim = torch.optim.Adam(self.params, lr=learning_rate, eps=epsilon)

    def learn_model(self):
        e_losses = []
        mu_e_losses = []
        errors = []
        mu_errors = []
        r_losses = []
        n_batches = []
        for epoch in range(1, self.n_train_epochs + 1):
            e_losses.append([])
            mu_e_losses.append([])
            r_losses.append([])
            errors.append([])
            mu_errors.append([])
            n_batches.append(0)

            for (states, mus, actions, rewards, smirls, combined_rewards, deltas, next_mus) \
                    in self.buffer.get_train_batches(self.batch_size):

                self.ensemble.train()
                self.reward_model.train()

                self.optim.zero_grad()
                # NOTE: DO I USE THIS FUNCTION IN NON-BATCH SITUATIONS?

                # LEARN MODEL
                e_loss, mu_e_loss, err, mu_err = self.ensemble.loss(states, mus, actions, deltas, next_mus, batch=True)
                r_loss = self.reward_model.loss(states, mus, actions, rewards, smirls, combined_rewards, batch=True)

                (e_loss + mu_e_loss + r_loss).backward()
                torch.nn.utils.clip_grad_norm_(
                    self.params, self.grad_clip_norm, norm_type=2
                )
                self.optim.step()

                e_losses[epoch - 1].append(e_loss.item())
                mu_e_losses[epoch - 1].append(mu_e_loss.item())
                r_losses[epoch - 1].append(r_loss.item())
                errors[epoch - 1].append(err.item())
                mu_errors[epoch - 1].append(mu_err.item())
                n_batches[epoch - 1] += 1

            if self.logger is not None and epoch % 20 == 0:
                avg_e_loss = self._get_avg_loss(e_losses, n_batches, epoch)
                avg_r_loss = self._get_avg_loss(r_losses, n_batches, epoch)
                avg_mu_loss = self._get_avg_loss(mu_e_losses, n_batches, epoch)
                message = "> Train epoch {} [ensemble {:.2f} | mean {:.2f} | reward {:.2f}]"
                self.logger.log(message.format(epoch, avg_e_loss, avg_mu_loss, avg_r_loss))

        return (
            self._get_avg_loss(e_losses, n_batches, epoch),
            self._get_avg_loss(mu_e_losses, n_batches, epoch),
            self._get_avg_loss(r_losses, n_batches, epoch),
            self._get_avg_loss(errors, n_batches, epoch),
            self._get_avg_loss(mu_errors, n_batches, epoch),
        )

    def reset_models(self):
        self.ensemble.reset_parameters()
        self.reward_model.reset_parameters()
        self.params = list(self.ensemble.parameters()) + list(
            self.reward_model.parameters()
        )
        self.optim = torch.optim.Adam(
            self.params, lr=self.learning_rate, eps=self.epsilon
        )

    def _get_avg_loss(self, losses, n_batches, epoch):
        epoch_loss = [sum(loss) / n_batch for loss, n_batch in zip(losses, n_batches)]
        return sum(epoch_loss) / epoch

import torch
import torch.nn.functional as F
from torchvision.utils import save_image

from tqdm import tqdm
import random
import os
import matplotlib.pyplot as plt
import seaborn as sns
import time

from minigpt_utils import prompt_wrapper, generator

def normalize(images):
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).cuda()
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).cuda()
    images = (images - mean[None, :, None, None]) / std[None, :, None, None]
    return images

def denormalize(images):
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).cuda()
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).cuda()
    images = images * std[None, :, None, None] + mean[None, :, None, None]
    return images



class Attacker:
    def __init__(self, args, model, targets, device='cuda:0', is_rtp=False):
        self.args = args
        self.model = model
        self.device = device
        self.is_rtp = is_rtp
        self.targets = targets
        self.num_targets = len(targets)
        self.loss_buffer = []
        self.model.eval()
        self.model.requires_grad_(False)

    def attack_loss(self, prompts, targets):

        context_embs = prompts.context_embs

        if len(context_embs) == 1:
            context_embs = context_embs * len(targets)  # expand to fit the batch_size

        assert len(context_embs) == len(
            targets), f"Unmathced batch size of prompts and targets {len(context_embs)} != {len(targets)}"

        batch_size = len(targets)
        self.model.llama_tokenizer.padding_side = "right"

        to_regress_tokens = self.model.llama_tokenizer(
            targets,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.model.max_txt_len,
            add_special_tokens=False
        ).to(self.device)
        to_regress_embs = self.model.llama_model.model.embed_tokens(to_regress_tokens.input_ids)

        bos = torch.ones([1, 1],
                         dtype=to_regress_tokens.input_ids.dtype,
                         device=to_regress_tokens.input_ids.device) * self.model.llama_tokenizer.bos_token_id
        bos_embs = self.model.llama_model.model.embed_tokens(bos)

        pad = torch.ones([1, 1],
                         dtype=to_regress_tokens.input_ids.dtype,
                         device=to_regress_tokens.input_ids.device) * self.model.llama_tokenizer.pad_token_id
        pad_embs = self.model.llama_model.model.embed_tokens(pad)

        T = to_regress_tokens.input_ids.masked_fill(
            to_regress_tokens.input_ids == self.model.llama_tokenizer.pad_token_id, -100
        )

        pos_padding = torch.argmin(T, dim=1)  # a simple trick to find the start position of padding

        input_embs = []
        targets_mask = []

        target_tokens_length = []
        context_tokens_length = []
        seq_tokens_length = []

        for i in range(batch_size):

            pos = int(pos_padding[i])
            if T[i][pos] == -100:
                target_length = pos
            else:
                target_length = T.shape[1]

            targets_mask.append(T[i:i + 1, :target_length])
            input_embs.append(to_regress_embs[i:i + 1, :target_length])  # omit the padding tokens

            context_length = context_embs[i].shape[1]
            seq_length = target_length + context_length

            target_tokens_length.append(target_length)
            context_tokens_length.append(context_length)
            seq_tokens_length.append(seq_length)

        max_length = max(seq_tokens_length)

        attention_mask = []

        for i in range(batch_size):
            # masked out the context from loss computation
            context_mask = (
                torch.ones([1, context_tokens_length[i] + 1],
                           dtype=torch.long).to(self.device).fill_(-100)  # plus one for bos
            )

            # padding to align the length
            num_to_pad = max_length - seq_tokens_length[i]
            padding_mask = (
                torch.ones([1, num_to_pad],
                           dtype=torch.long).to(self.device).fill_(-100)
            )

            targets_mask[i] = torch.cat([context_mask, targets_mask[i], padding_mask], dim=1)
            input_embs[i] = torch.cat([bos_embs, context_embs[i], input_embs[i],
                                       pad_embs.repeat(1, num_to_pad, 1)], dim=1)
            attention_mask.append(torch.LongTensor([[1] * (1 + seq_tokens_length[i]) + [0] * num_to_pad]))

        targets = torch.cat(targets_mask, dim=0).to(self.device)
        inputs_embs = torch.cat(input_embs, dim=0).to(self.device)
        attention_mask = torch.cat(attention_mask, dim=0).to(self.device)

        outputs = self.model.llama_model(
            inputs_embeds=inputs_embs,
            attention_mask=attention_mask,
            return_dict=True,
            labels=targets,
        )
        loss = outputs.loss

        return loss

    def plot_loss(self):
        sns.set_theme()
        num_iters = len(self.loss_buffer)

        x_ticks = list(range(0, num_iters))

        # Plot and label the training and validation loss values
        plt.plot(x_ticks, self.loss_buffer, label='Target Loss')

        # Add in a title and axes labels
        plt.title('Loss Plot')
        plt.xlabel('Iters')
        plt.ylabel('Loss')

        # Display the plot
        plt.legend(loc='best')
        plt.savefig('%s/loss_curve.png' % (self.args.save_dir))
        plt.clf()

        torch.save(self.loss_buffer, '%s/loss' % (self.args.save_dir))

        # def linf_clamp_(self, dx, x, eps):
        #     """Clamps perturbation `dx` to fit L_inf norm and image bounds.
        #
        #     Limit the L_inf norm of `dx` to be <= `eps`, and the bounds of `x + dx`
        #     to be in `[clip_min, clip_max]`.
        #
        #     Return: the clamped perturbation `dx`.
        #     """
        #
        #     # dx_clamped = self.batch_clamp(eps, dx)
        #     dx_clamped = torch.clamp(dx, min=-eps, max=eps)
        #     # x_adv = self.clamp(x + dx_clamped, clip_min, clip_max)
        #     x_adv = torch.clamp(x + dx_clamped, min=0, max=1)
        #     dx += x_adv - x - dx
        #     return dx

    def unlinf_clamp_(self, dx, x):
        """Clamps perturbation `dx` to fit image bounds.
        The bounds of `x + dx` will be in `[0, 1]`.
        Return: the clamped perturbation `dx`.
        """

        # 只对 x + dx 进行裁剪，将结果限制在 [0, 1] 范围内
        x_adv = torch.clamp(x + dx, min=0, max=1)

        # 原地更新 `dx`，使其保持对抗扰动
        dx += x_adv - x - dx
        return dx

    # def compute_spsa_gradient(self, img, text_prompts, batch_targets, batch_size, scaler=1):
    #     print("Calculating SPSA gradient optimizer...")
    #
    #     # =========================================
    #     # 选择对称分布是收敛的重要条件之一
    #     # 学习率和扰动幅度的适当选择对应实现良好的收敛至关重要
    #     # 目标函数在最优点的二阶和三阶导数需要非零以保证稳定的收敛
    #     # 生成随机扰动每个元素为 -1 或 1
    #     # delta = torch.randint(0, 2, img.shape, dtype=img.dtype, device=img.device) * 2 - 1
    #     # 每个元素为 -1 或 1
    #     # delta = torch.bernoulli(torch.full(img.shape, 0.5, device=img.device)) * 2 - 1
    #     delta = (torch.bernoulli(torch.full(img.shape, 0.5, device=img.device)) * 2 - 1) * 0.01
    #     # =========================================
    #
    #     # 创建正向和负向扰动的图像
    #     img_plus = img + scaler * delta
    #     img_minus = img - scaler * delta
    #
    #     # 计算正向和负向扰动的损失
    #     with torch.no_grad():
    #         img_plus = normalize(img_plus)
    #         prompt_plus = prompt_wrapper.Prompt(model=self.model, text_prompts=text_prompts, img_prompts=[[img_plus]])
    #         prompt_plus.img_embs = prompt_plus.img_embs * batch_size
    #         prompt_plus.update_context_embs()
    #         loss_plus = self.attack_loss(prompt_plus, batch_targets)
    #
    #         img_minus = normalize(img_minus)
    #         prompt_minus = prompt_wrapper.Prompt(model=self.model, text_prompts=text_prompts, img_prompts=[[img_minus]])
    #         prompt_minus.img_embs = prompt_minus.img_embs * batch_size
    #         prompt_minus.update_context_embs()
    #         loss_minus = self.attack_loss(prompt_minus, batch_targets)
    #
    #     # 计算梯度估计
    #     grad_estimate = (loss_plus - loss_minus) / (2 * scaler * delta)
    #
    #     return grad_estimate

    # def compute_spsa_gradient(self, img, text_prompts, batch_targets, batch_size, num_samples=1, scaler=0.01):
    #     print("Calculating SPSA gradient...")
    #
    #     # 初始化梯度估计矩阵
    #     grad_estimate = torch.zeros_like(img).to(self.device)
    #
    #     # 多次采样计算梯度
    #     for _ in range(num_samples):
    #         # 随机生成扰动 delta
    #         # delta = (torch.bernoulli(torch.full(img.shape, 0.5, device=img.device)) * 2 - 1)
    #         # # # 创建正向和负向扰动的图像
    #         # img_plus = img + scaler * delta
    #         # img_minus = img - scaler * delta
    #
    #         # delta = torch.rand_like(img) * (1e-3 - 1e-4) + 1e-4
    #         delta = torch.rand_like(img) * (1e-4 - 1e-5) + 1e-5
    #         # delta = torch.rand_like(img) * (1e-7 - 1e-8) + 1e-8
    #         img_plus = img + delta
    #         img_minus = img - delta
    #
    #         # 计算正向和负向扰动的损失
    #         with torch.no_grad():
    #             # 处理正向扰动的图像
    #             img_plus_norm = normalize(img_plus)
    #             prompt_plus = prompt_wrapper.Prompt(model=self.model, text_prompts=text_prompts,
    #                                                 img_prompts=[[img_plus_norm]])
    #             prompt_plus.img_embs = prompt_plus.img_embs * batch_size
    #             prompt_plus.update_context_embs()
    #             loss_plus = self.attack_loss(prompt_plus, batch_targets)
    #
    #             # 处理负向扰动的图像
    #             img_minus_norm = normalize(img_minus)
    #             prompt_minus = prompt_wrapper.Prompt(model=self.model, text_prompts=text_prompts,
    #                                                  img_prompts=[[img_minus_norm]])
    #             prompt_minus.img_embs = prompt_minus.img_embs * batch_size
    #             prompt_minus.update_context_embs()
    #             loss_minus = self.attack_loss(prompt_minus, batch_targets)
    #
    #             # 累加每次采样的梯度估计
    #             grad_estimate += (loss_plus - loss_minus) / (2 * scaler * delta)
    #             # grad_estimate += (loss_plus - loss_minus) / (2 * delta)
    #
    #     # 对所有采样结果取平均
    #     grad_estimate /= num_samples
    #
    #     return grad_estimate


    # def attack_unconstrained_spsa(self, text_prompt, img, batch_size=8, num_iter=2000, alpha=1 / 255):
    #     """ Unconstrained adversarial attack """
    #     print('>>> batch_size:', batch_size)
    #     my_generator = generator.Generator(model=self.model)
    #
    #     # 初始化图片
    #     adv_noise = torch.rand_like(img).to(self.device)  # [0,1]
    #
    #     # # 初始化扰动 dx，初始为全零
    #     # dx = torch.zeros_like(img).to(self.device)
    #     # dx.grd = torch.zeros_like(dx)
    #     # # 使用 Adam 优化器对 dx 进行优化 lr=1e-4
    #     # optimizer = torch.optim.Adam([dx], lr=0.01)
    #
    #     for t in tqdm(range(num_iter + 1)):
    #         batch_targets = random.sample(self.targets, batch_size)
    #         text_prompts = [text_prompt] * batch_size
    #
    #         # ================================================
    #         # optimizer.zero_grad()  # 清零优化器中的梯度
    #         # Calculate SPSA gradient
    #         # dx.grad  = self.compute_spsa_gradient(x_adv, text_prompts, batch_targets, batch_size)
    #         # # dx.grad = spsa_gradient.detach() / spsa_gradient.norm(p=1)
    #         # # 使用优化器更新 dx
    #         # optimizer.step()
    #         # dx = self.unlinf_clamp_(dx, x_adv)
    #         # # print(dx)
    #         # # 更新 adv_noise，并将其裁剪到合法范围
    #         # adv_noise.data = torch.clamp(adv_noise.data + dx, 0, 1)
    #         # ================================================
    #
    #         # ================================================
    #         # Calculate SPSA gradient
    #         spsa_grad = self.compute_spsa_gradient(adv_noise, text_prompts, batch_targets, batch_size, num_samples=1, scaler=0.01)
    #         adv_noise.data = (adv_noise.data - alpha * spsa_grad.detach().sign()).clamp(0, 1)
    #         # ================================================
    #
    #         x_adv = normalize(adv_noise)
    #         prompt = prompt_wrapper.Prompt(model=self.model, text_prompts=text_prompts, img_prompts=[[x_adv]])
    #         prompt.img_embs = prompt.img_embs * batch_size
    #         prompt.update_context_embs()
    #
    #         # Calculate the target loss using the attack loss function
    #         target_loss = self.attack_loss(prompt, batch_targets)
    #         self.loss_buffer.append(target_loss.item())
    #         print("target_loss: %f" % (target_loss.item()))
    #
    #         if t % 20 == 0:
    #             self.plot_loss()
    #
    #         # Output results every 100 iterations
    #         if t % 100 == 0:
    #             print('######### Output - Iter = %d ##########' % t)
    #             x_adv = normalize(adv_noise)
    #             prompt.update_img_prompts([[x_adv]])
    #             prompt.img_embs = prompt.img_embs * batch_size
    #             prompt.update_context_embs()
    #             with torch.no_grad():
    #                 response, _ = my_generator.generate(prompt)
    #             print('>>>', response)
    #
    #             adv_img_prompt = denormalize(x_adv).detach().cpu()
    #             adv_img_prompt = adv_img_prompt.squeeze(0)
    #             save_image(adv_img_prompt, '%s/bad_prompt_temp_%d.bmp' % (self.args.save_dir, t))
    #
    #     return adv_img_prompt

    def compute_spsa_gradient(self, img, text_prompts, batch_targets, batch_size, scaler):
        print("======Calculating SPSA gradient=======")

        delta = torch.bernoulli(torch.full(img.shape, 0.5, device=img.device, requires_grad=False)) * 2 - 1
        img_plus = img + scaler * delta
        img_minus = img - scaler * delta

        with torch.no_grad():
            img_plus_norm = normalize(img_plus)
            prompt_plus = prompt_wrapper.Prompt(model=self.model, text_prompts=text_prompts,
                                                img_prompts=[[img_plus_norm]])
            prompt_plus.img_embs = prompt_plus.img_embs * batch_size
            prompt_plus.update_context_embs()
            loss_plus = self.attack_loss(prompt_plus, batch_targets)

            img_minus_norm = normalize(img_minus)
            prompt_minus = prompt_wrapper.Prompt(model=self.model, text_prompts=text_prompts,
                                                 img_prompts=[[img_minus_norm]])
            prompt_minus.img_embs = prompt_minus.img_embs * batch_size
            prompt_minus.update_context_embs()
            loss_minus = self.attack_loss(prompt_minus, batch_targets)

        grad_estimate = (loss_plus - loss_minus) / (2 * scaler * delta)
        return grad_estimate

        # # Apply momentum to smooth gradient updates
        # if momentum is None:
        #     momentum = grad_estimate
        # else:
        #     momentum = momentum_beta * self.momentum + (1 - momentum_beta) * grad_estimate
        #
        # return self.momentum

    def attack_unconstrained_spsa(self, text_prompt, img, batch_size=8, num_iter=1000, alpha=1 / 255, scaler_init=0.01, patience=1000):
        print('>>> batch_size:', batch_size)
        my_generator = generator.Generator(model=self.model)
        adv_noise = torch.rand_like(img).to(self.device).clone()

        scaler = scaler_init
        best_loss = float('inf')
        patience_counter = 0

        for t in tqdm(range(num_iter + 1)):
            batch_targets = random.sample(self.targets, batch_size)
            text_prompts = [text_prompt] * batch_size
            x_adv = normalize(adv_noise)

            prompt = prompt_wrapper.Prompt(model=self.model, text_prompts=text_prompts, img_prompts=[[x_adv]])
            prompt.img_embs = prompt.img_embs * batch_size
            prompt.update_context_embs()
            target_loss = self.attack_loss(prompt, batch_targets)

            # Update best loss and reset patience counter if new minimum loss is found
            # if target_loss.item() < best_loss:
            #     best_loss = target_loss.item()
            #     patience_counter = 0
            # else:
            #     patience_counter += 1
            #
            # # If patience limit is reached, stop early
            # if patience_counter >= patience:
            #     print(f"Early stopping at iteration {t} with best loss {best_loss}")
            #     break

            spsa_gradient = self.compute_spsa_gradient(adv_noise, text_prompts, batch_targets, batch_size, scaler)
            adv_noise.data = (adv_noise.data - alpha * spsa_gradient.detach().sign()).clamp(0, 1)

            # Adaptive step size to encourage convergence
            # if t > 0 and target_loss.item() > self.loss_buffer[-1]:
            #     alpha = max(alpha * 0.5, 1 / 255)  # Reduce alpha, but ensure it does not go below 1/255, the minimum step size allowed
            # else:
            #     alpha = min(alpha * 1.1, 0.1)  # Increase alpha slightly but keep it within a reasonable range

            self.loss_buffer.append(target_loss.item())
            print("target_loss: %f" % (target_loss.item()))

            if t % 20 == 0:
                self.plot_loss()

            if t % 100 == 0:
                print('######### Output - Iter = %d ##########' % t)
                x_adv = normalize(adv_noise)
                prompt.update_img_prompts([[x_adv]])
                prompt.img_embs = prompt.img_embs * batch_size
                prompt.update_context_embs()
                with torch.no_grad():
                    response, _ = my_generator.generate(prompt)
                print('>>>', response)

                adv_img_prompt = denormalize(x_adv).detach().cpu()
                adv_img_prompt = adv_img_prompt.squeeze(0)
                save_image(adv_img_prompt, '%s/bad_prompt_temp_%d.bmp' % (self.args.save_dir, t))

        return adv_img_prompt
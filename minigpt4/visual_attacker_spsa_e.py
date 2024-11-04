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

    def average_compute_spsa_gradient(self, img, text_prompts, batch_targets, batch_size):
        print("Calculating SPSA gradient...")

        grad_sum = 0
        for i in range(10):
            # 生成一个与 x 形状相同的随机噪声矩阵，值在 0 到 1e-2之间
            # delta = torch.rand_like(img) * 1e-2
            # delta = torch.rand_like(img) * (1e-3 - 1e-4) + 1e-4

            # delta = torch.randn_like(img) * 1e-4
            # [0.0001, 0.0001]
            delta = torch.randn_like(img) * (1e-4 - 1e-5) + 1e-5
            # [0.00001, 0.0001]

            img_plus = img + delta
            img_minus = img - delta

            # 计算正向和负向扰动的损失
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

            # 计算梯度估计
            grad_sum += (loss_plus - loss_minus) / (2 * delta)
        grad_estimate = grad_sum / 10

        return grad_estimate

    # def compute_spsa_gradient(self, img, text_prompts, batch_targets, batch_size):
    #     print("=====Calculating SPSA gradient=====")
    #
    #     #=======生成一个与 x 形状相同的随机噪声矩阵，值在 0 到 1e-2之间 [0, 0.01]
    #     # delta = torch.rand_like(img) * 1e-2
    #     # delta = torch.rand_like(img) * (1e-3 - 1e-4) + 1e-4
    #
    #     #====== normal distribution
    #     # delta = torch.randn_like(img) * 1e-4
    #     # delta = torch.randn_like(img) * 1e-5
    #     delta = torch.randn_like(img) * (1e-4 - 1e-5) + 1e-5
    #     img_plus = img + delta
    #     img_minus = img - delta
    #
    #     max_diff = 1e-2

    #     with torch.no_grad():
    #         img_plus_norm = normalize(img_plus)
    #         prompt_plus = prompt_wrapper.Prompt(model=self.model, text_prompts=text_prompts,
    #                                             img_prompts=[[img_plus_norm]])
    #         prompt_plus.img_embs = prompt_plus.img_embs * batch_size
    #         prompt_plus.update_context_embs()
    #         loss_plus = self.attack_loss(prompt_plus, batch_targets)
    #         print("loss_plus=========", loss_plus)
    #
    #         img_minus_norm = normalize(img_minus)
    #         prompt_minus = prompt_wrapper.Prompt(model=self.model, text_prompts=text_prompts,
    #                                              img_prompts=[[img_minus_norm]])
    #         prompt_minus.img_embs = prompt_minus.img_embs * batch_size
    #         prompt_minus.update_context_embs()
    #         loss_minus = self.attack_loss(prompt_minus, batch_targets)
    #         print("loss_minus========", loss_minus)
    #
    #         dloss = loss_plus - loss_minus
    #         print("dloss===========", dloss)
    #         if max_diff is not None and abs(dloss) > max_diff:
    #             dloss = max_diff * (1 if dloss > 0 else -1)
    #         print("dloss new values=========", dloss)
    #     # 计算梯度估计
    #     grad_estimate = dloss / (2 * delta)
    #
    #     return grad_estimate

    def compute_spsa_gradient(self, img, text_prompts, batch_targets, batch_size):
        print("=====Calculating SPSA gradient=====")

        # =======生成一个与 x 形状相同的随机噪声矩阵，值在 0 到 1e-2之间 [0, 0.01]
        # delta = torch.rand_like(img) * 1e-2
        # delta = torch.rand_like(img) * (1e-3 - 1e-4) + 1e-4

        # ====== normal distribution
        delta = torch.randn_like(img) * 1e-4

        # 计算正向和负向扰动的损失
        with torch.no_grad():
            img_plus_norm = normalize(img)
            img_plus = img_plus_norm + delta
            prompt_plus = prompt_wrapper.Prompt(model=self.model, text_prompts=text_prompts,
                                                img_prompts=[[img_plus]])
            prompt_plus.img_embs = prompt_plus.img_embs * batch_size
            prompt_plus.update_context_embs()
            loss_plus = self.attack_loss(prompt_plus, batch_targets)
            # print("loss_plus=========", loss_plus)

            img_minus_norm = normalize(img)
            img_minus = img_minus_norm + delta
            prompt_minus = prompt_wrapper.Prompt(model=self.model, text_prompts=text_prompts,
                                                 img_prompts=[[img_minus]])
            prompt_minus.img_embs = prompt_minus.img_embs * batch_size
            prompt_minus.update_context_embs()
            loss_minus = self.attack_loss(prompt_minus, batch_targets)
            # print("loss_minus========", loss_minus)

            dloss = loss_plus - loss_minus
            print("dloss=======", dloss)
        grad_estimate = dloss / (2 * delta)
        print("grad_estimate======", grad_estimate)

        return grad_estimate

    def attack_unconstrained_spsa_SignSGD(self, text_prompt, img, batch_size=8, num_iter=2000, alpha=1 / 255):
        """ Unconstrained adversarial attack """
        print('>>> batch_size:', batch_size)
        my_generator = generator.Generator(model=self.model)
        adv_noise = torch.rand_like(img).to(self.device).clone() # [0,1]

        for t in tqdm(range(num_iter + 1)):
            #  TODO 学习率衰减要怎么设置 因为alpha已经是像素最小值
            # alpha = 0.01 / torch.sqrt(torch.tensor(num_iter + 1))

            batch_targets = random.sample(self.targets, batch_size)
            text_prompts = [text_prompt] * batch_size
            x_adv = normalize(adv_noise)

            prompt = prompt_wrapper.Prompt(model=self.model, text_prompts=text_prompts, img_prompts=[[x_adv]])
            prompt.img_embs = prompt.img_embs * batch_size
            prompt.update_context_embs()
            target_loss = self.attack_loss(prompt, batch_targets)

            # ================================================
            # Calculate SPSA gradient
            spsa_gradient = self.compute_spsa_gradient(adv_noise, text_prompts, batch_targets, batch_size)
            # print("adv_noise======", adv_noise)
            adv_noise.data = (adv_noise.data - alpha * spsa_gradient.detach().sign()).clamp(0, 1)
            # ================================================

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

    def attack_constrained_spsa_SignSGD(self, text_prompt, img, batch_size = 8, num_iter=2000, alpha=1/255, epsilon = 128/255 ):
        print('>>> batch_size:', batch_size)

        my_generator = generator.Generator(model=self.model)

        adv_noise = torch.rand_like(img).to(self.device) * 2 * epsilon - epsilon
        x = denormalize(img).clone().to(self.device)
        adv_noise.data = (adv_noise.data + x.data).clamp(0, 1) - x.data


        for t in tqdm(range(num_iter + 1)):
            batch_targets = random.sample(self.targets, batch_size)
            text_prompts = [text_prompt] * batch_size

            x_adv = x + adv_noise
            x_adv = normalize(x_adv)

            prompt = prompt_wrapper.Prompt(model=self.model, text_prompts=text_prompts, img_prompts=[[x_adv]])
            prompt.img_embs = prompt.img_embs * batch_size
            prompt.update_context_embs()
            target_loss = self.attack_loss(prompt, batch_targets)

            # ================================================
            # Calculate SPSA gradient
            spsa_gradient = self.compute_spsa_gradient(adv_noise, text_prompts, batch_targets, batch_size)
            # ================================================

            adv_noise.data = (adv_noise.data - alpha * spsa_gradient.detach().sign()).clamp(-epsilon, epsilon)
            adv_noise.data = (adv_noise.data + x.data).clamp(0, 1) - x.data

            self.loss_buffer.append(target_loss.item())

            print("target_loss: %f" % (
                target_loss.item())
                  )

            if t % 20 == 0:
                self.plot_loss()

            if t % 100 == 0:
                print('######### Output - Iter = %d ##########' % t)
                x_adv = x + adv_noise
                x_adv = normalize(x_adv)
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


    def attack_unconstrained_spsa_Adam(self, text_prompt, img, batch_size=8, num_iter=2000, alpha=1 / 255,
                                        beta_1=0.9, beta_2=0.999, epsilon=1e-8):
        """ Unconstrained adversarial attack using AdaMM """
        print('>>> batch_size:', batch_size)
        print("Unconstrained adversarial attack using AdaMM")
        my_generator = generator.Generator(model=self.model)
        adv_noise = torch.rand_like(img).to(self.device).clone()  # [0,1]

        # Initialize moment estimates
        m = torch.zeros_like(adv_noise).to(self.device)
        v = torch.zeros_like(adv_noise).to(self.device)


        for t in tqdm(range(1, num_iter + 1)):
            batch_targets = random.sample(self.targets, batch_size)
            text_prompts = [text_prompt] * batch_size
            x_adv = normalize(adv_noise)

            prompt = prompt_wrapper.Prompt(model=self.model, text_prompts=text_prompts, img_prompts=[[x_adv]])
            prompt.img_embs = prompt.img_embs * batch_size
            prompt.update_context_embs()
            target_loss = self.attack_loss(prompt, batch_targets)

            # ================================================
            # Calculate SPSA gradient
            spsa_gradient = self.compute_spsa_gradient(adv_noise, text_prompts, batch_targets, batch_size)
            # ================================================

            # Update biased first moment estimate
            m = beta_1 * m + (1 - beta_1) * spsa_gradient

            # Update biased second raw moment estimate
            v = beta_2 * v + (1 - beta_2) * (spsa_gradient * spsa_gradient)

            # Compute bias-corrected first moment estimate
            m_hat = m / (1 - beta_1 ** t)

            # Compute bias-corrected second moment estimate
            v_hat = v / (1 - beta_2 ** t)

            # Update adversarial noise
            adv_noise.data = (adv_noise.data - alpha * m_hat / (torch.sqrt(v_hat) + epsilon)).clamp(0, 1)

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





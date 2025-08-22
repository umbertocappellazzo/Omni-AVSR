#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  4 14:49:14 2025

@author: umbertocappellazzo
"""

import torch.nn.functional as F
import torch
from typing import Tuple

def compute_dpo_loss(self,
                     policy_chosen_log_probs: torch.FloatTensor,
                     policy_rejected_log_probs: torch.FloatTensor,
                     reference_chosen_log_probs: torch.FloatTensor,
                     reference_rejected_log_probs: torch.FloatTensor, 
                     beta = 0.1) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        
    """Compute the DPO loss for a batch of policy and reference model log probabilities.

     Args:
         policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
         policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
         reference_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
         reference_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)

     Returns:
         A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
         The losses tensor contains the DPO loss for each example in the batch.
         The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
     """   
    
    policy_log_ratio = policy_chosen_log_probs - policy_rejected_log_probs
    
    reference_log_ratio = reference_chosen_log_probs - reference_rejected_log_probs
    
    logits = policy_log_ratio - reference_log_ratio
    
    dpo_loss = -F.logsigmoid(beta*logits).mean()
    
    # Additional metrics for monitoring
    chosen_rewards = beta *(policy_chosen_log_probs-reference_chosen_log_probs).detach()
    rejected_rewards = beta*(policy_rejected_log_probs-reference_rejected_log_probs).detach()
    
    reward_accuracy = (chosen_rewards > rejected_rewards).float().mean()
    
    return (dpo_loss, chosen_rewards.mean().item(), rejected_rewards.mean().item(), 
            reward_accuracy.item(), (chosen_rewards - rejected_rewards).mean().item()
            )



def get_batch_logps(
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        label_pad_token_id: int = -100,
    ) -> torch.FloatTensor:
        """Compute the log probabilities of the given labels under the given logits.

        Args:
            logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
            labels: Labels for which to compute the log probabilities. Label tokens with a value of label_pad_token_id are ignored. Shape: (batch_size, sequence_length)
            label_pad_token_id: The label pad token id.
            
        Returns:
            A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
        """
        if logits.shape[:-1] != labels.shape:
            raise ValueError("Logits (batch and sequence length dim) and labels must have the same shape.")

        labels = labels[:, 1:].clone()
        logits = logits[:, :-1, :]
        loss_mask = labels != label_pad_token_id

        # dummy token; we'll ignore the losses on these tokens later
        labels[labels == label_pad_token_id] = 0

        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

        return (per_token_logps * loss_mask).sum(-1)
        
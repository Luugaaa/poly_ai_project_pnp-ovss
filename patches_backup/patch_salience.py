import re

with open("core/salience_dropout.py", "r") as f:
    text = f.read()

# Replace the loop variables
text = text.replace("total_passes = 1 + dropout_rounds", "total_passes = dropout_rounds")

# In the accumulation block, double weight the first pass if t == 0:
acc_block = """
        for c, t_idx in class_token_indices.items():
            flat_scores = compute_gradcam_salience(
                attn, attn_grad, head_idx, t_idx
            )                                   # [P*P]
            seg_scores = strategy.aggregate(flat_scores)  # [N]
            for idx in dropped:
                seg_scores[idx] = 0.0
            class_seg_scores[c] = seg_scores
            
            weight = 2.0 if t == 0 else 1.0
            accumulated[c] = accumulated[c] + (seg_scores * weight)
            sum_scores = sum_scores + seg_scores
"""

# Find the block where seg_scores is computed and accumulated
pattern = re.compile(
    r"        for c, t_idx in class_token_indices\.items\(\):.*?"
    r"            accumulated\[c\] = accumulated\[c\] \+ seg_scores\s*"
    r"            sum_scores = sum_scores \+ seg_scores",
    re.DOTALL
)

text = pattern.sub(acc_block.strip() + "\n", text)

with open("core/salience_dropout.py", "w") as f:
    f.write(text)

print("Patch applied to salience_dropout.py")

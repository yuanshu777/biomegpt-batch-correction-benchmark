# scGPT Batch Correction Reference Notes

Local source reference before packaging:

```text
C:\Users\Yuanshu\Desktop\Ali lab\scgpt\model\scGPT-main
```

Key files in scGPT:

```text
examples/finetune_integration.py
scgpt/model/model.py
scgpt/model/grad_reverse.py
```

Important scGPT snippets/ideas:

```python
from scgpt.model import TransformerModel, AdversarialDiscriminator
```

```python
adata.obs['str_batch'] = adata.obs[ori_batch_col].astype(str)
batch_id_labels = adata.obs['str_batch'].astype('category').cat.codes.values
adata.obs['batch_id'] = batch_id_labels
```

```python
model = TransformerModel(
    ..., do_dab=True,
    use_batch_labels=True,
    num_batch_labels=num_batch_types,
    domain_spec_batchnorm=DSBN,
)
```

```python
loss_dab = criterion_dab(output_dict['dab_output'], batch_labels)
loss = loss + config.dab_weight * loss_dab
```

Gradient reversal implementation:

```python
class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambd, None
```

Adversarial discriminator concept:

```python
class AdversarialDiscriminator(nn.Module):
    def forward(self, x):
        if self.reverse_grad:
            x = grad_reverse(x, lambd=1.0)
        return self.out_layer(mlp_layers(x))
```

BiomeGPT translation:

- gene token -> species token
- cell embedding -> sample prompt / CLS embedding
- batch column -> external study/cohort/batch label
- DAB output -> batch logits from sample prompt
- goal -> reduce batch predictability while preserving phenotype predictability

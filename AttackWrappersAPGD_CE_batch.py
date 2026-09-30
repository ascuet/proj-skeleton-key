import torch
import torch.nn.functional as F

from Utilities import DataManagerPytorch as DMP

try:
    from spikingjelly.clock_driven import functional
except ImportError:
    functional = None


def projection_linf(x_adv, x_orig, eps):
    """Project x_adv into the L-infinity epsilon-ball around x_orig."""
    diff = torch.clamp(x_adv - x_orig, min=-eps, max=eps)
    return x_orig + diff


def _forward_logits(model, x, modelPlus=None):
    """
    Forward pass that preserves the special SNN behavior already used by
    AutoAttackNativePytorch / FGSMNativeGradient in the P_SAGA file.

    For ordinary models, this is simply model(x).
    """
    model_name = getattr(modelPlus, "modelName", "") if modelPlus is not None else ""

    if model_name == "SNN ResNet Backprop":
        if functional is None:
            raise ImportError(
                "spikingjelly is required for SNN ResNet Backprop attacks."
            )
        functional.reset_net(model)
        output = model(x)
        return output.mean(0)

    return model(x)


def get_grad_ce_batch(model, x, y, modelPlus=None):
    """
    Compute input gradients for the whole batch in one autograd call.

    CE is calculated per sample (reduction='none'). Summing the per-sample
    losses is mathematically valid here because each sample's loss depends
    only on its own input.
    """
    x_in = x.detach().clone().requires_grad_(True)

    logits = _forward_logits(model, x_in, modelPlus)
    if logits.ndim != 2:
        raise RuntimeError(
            "CE APGD expects model logits with shape [batch, classes]. "
            f"Got shape {tuple(logits.shape)}. Check the model-specific forward adapter."
        )

    losses = F.cross_entropy(logits, y, reduction="none")

    grad = torch.autograd.grad(
        losses.sum(),
        x_in,
        retain_graph=False,
        create_graph=False,
    )[0]

    # Match the historical SAGA handling for SNN VGG if its input has a
    # temporal dimension. For ordinary image models this is a no-op.
    if modelPlus is not None and getattr(modelPlus, "modelName", "") == "SNN VGG-16 Backprop":
        if grad.ndim == 5:
            grad = grad.sum(-1)

    return grad.detach()


def get_ce_loss_batch(model, x, y, modelPlus=None):
    """Return per-sample CE losses for best-adversarial-example tracking."""
    logits = _forward_logits(model, x, modelPlus)
    if logits.ndim != 2:
        raise RuntimeError(
            "CE APGD expects model logits with shape [batch, classes]. "
            f"Got shape {tuple(logits.shape)}. Check the model-specific forward adapter."
        )
    return F.cross_entropy(logits, y, reduction="none")


def _build_checkpoints(num_steps):
    """Same checkpoint schedule used by the supplied batched APGD file."""
    checkpoints = [0]
    p_prev2, p_prev1 = 0.0, 0.22

    first = int(p_prev1 * num_steps)
    checkpoints.append(first)

    while checkpoints[-1] < num_steps:
        delta = max(p_prev1 - p_prev2 - 0.03, 0.06)
        p_next = p_prev1 + delta
        w_next = int(p_next * num_steps)
        # Protect against a duplicate checkpoint caused by integer rounding.
        if w_next <= checkpoints[-1]:
            w_next = checkpoints[-1] + 1
        checkpoints.append(w_next)
        p_prev2, p_prev1 = p_prev1, p_next

    return checkpoints


def APGDNativePytorch_CE(
    device,
    dataLoader,
    model,
    modelPlus,
    epsilonMax,
    etaStart,
    numSteps,
    clipMin,
    clipMax,
    targeted=False,
    alpha=0.75,
    rho=0.75,
    random_start=False,
):
    """
    Fast batched APGD using Cross-Entropy loss.

    Parameters mirror AutoAttackNativePytorch() where practical, with
    additional optional APGD controls at the end.

    Important:
      - The attack is fully batched: one forward/backward pass handles the
        complete mini-batch.
      - CE is computed with reduction='none', preserving per-image tracking.
      - modelPlus.formatDataLoader() is used so model-specific preprocessing
        remains consistent with the existing SAGA attack pipeline.
      - Targeted APGD is intentionally not implemented, matching the behavior
        of the existing AutoAttackNativePytorch() implementation.
    """
    if targeted:
        raise ValueError("APGDNativePytorch_CE currently supports only untargeted attacks.")
    if numSteps < 1:
        raise ValueError("numSteps must be >= 1.")
    if epsilonMax < 0:
        raise ValueError("epsilonMax must be >= 0.")
    if etaStart <= 0:
        raise ValueError("etaStart must be > 0.")

    model = model.to(device)
    model.eval()

    # Keep the same model-specific formatting used by the existing
    # AutoAttackNativePytorch() implementation.
    if modelPlus is not None and hasattr(modelPlus, "formatDataLoader"):
        dataLoader = modelPlus.formatDataLoader(dataLoader)

    N = len(dataLoader.dataset)
    C, H, W = DMP.GetOutputShape(dataLoader)

    # Keep CPU output storage so GPU memory is reserved primarily for the
    # current attack batch rather than the complete dataset.
    x_adv_all = torch.empty(N, C, H, W, dtype=torch.float32)
    y_all = torch.empty(N, dtype=torch.long)

    W_checkpoints = _build_checkpoints(numSteps)
    idx_out = 0

    for x_clean, y in dataLoader:
        bs = x_clean.size(0)
        x_clean = x_clean.to(device, non_blocking=True).detach()
        y = y.to(device, dtype=torch.long, non_blocking=True)

        # Initial point.
        if random_start:
            delta = torch.empty_like(x_clean).uniform_(-epsilonMax, epsilonMax)
            x_k = torch.clamp(x_clean + delta, clipMin, clipMax).detach()
            x_k = projection_linf(x_k, x_clean, epsilonMax).detach()
        else:
            x_k = x_clean.clone()

        # Initial step.
        grad = get_grad_ce_batch(model, x_k, y, modelPlus)
        x_next = x_k + float(etaStart) * grad.sign()
        x_next = torch.clamp(x_next, clipMin, clipMax)
        x_next = projection_linf(x_next, x_clean, epsilonMax).detach()

        with torch.no_grad():
            f_x0 = get_ce_loss_batch(model, x_k, y, modelPlus)
            f_x1 = get_ce_loss_batch(model, x_next, y, modelPlus)

        better = f_x1 > f_x0
        x_best = x_k.clone()
        x_best[better] = x_next[better]
        f_best = torch.where(better, f_x1, f_x0)

        x_prev = x_k.clone()
        x_k = x_next.clone()

        # Per-sample step sizes. Keeping shape [B,1,1,1] makes broadcasting
        # fully vectorized and avoids a Python loop over samples.
        eta = torch.full(
            (bs, 1, 1, 1),
            float(etaStart),
            device=device,
            dtype=x_clean.dtype,
        )

        improvement = torch.zeros(bs, device=device, dtype=torch.int32)
        checkpoint_ptr = 1
        prev_eta = eta.clone()
        prev_f_best = f_best.clone()

        for k in range(1, numSteps):
            grad = get_grad_ce_batch(model, x_k, y, modelPlus)

            # APGD gradient/sign step.
            z_next = x_k + eta * grad.sign()
            z_next = torch.clamp(z_next, clipMin, clipMax)
            z_next = projection_linf(z_next, x_clean, epsilonMax)

            # Momentum/interpolation step used in the supplied batched APGD.
            x_next = (
                x_k
                + alpha * (z_next - x_k)
                + (1.0 - alpha) * (x_k - x_prev)
            )
            x_next = projection_linf(x_next, x_clean, epsilonMax)
            x_next = torch.clamp(x_next, clipMin, clipMax).detach()

            with torch.no_grad():
                f_k = get_ce_loss_batch(model, x_k, y, modelPlus)
                f_next = get_ce_loss_batch(model, x_next, y, modelPlus)

            improvement += (f_next > f_k).to(torch.int32)

            better = f_next > f_best
            x_best[better] = x_next[better]
            f_best[better] = f_next[better]

            # Adaptive step-size reduction.
            if checkpoint_ptr < len(W_checkpoints) and k == W_checkpoints[checkpoint_ptr]:
                interval = (
                    W_checkpoints[checkpoint_ptr]
                    - W_checkpoints[checkpoint_ptr - 1]
                )

                cond1 = improvement.to(torch.float32) < (rho * interval)
                same_eta = torch.isclose(eta, prev_eta).all(dim=1).all(dim=1).all(dim=1)
                cond2 = same_eta & torch.isclose(f_best, prev_f_best)
                reset_mask = cond1 | cond2

                if reset_mask.any():
                    eta[reset_mask] /= 2.0
                    x_next[reset_mask] = x_best[reset_mask]
                    x_k[reset_mask] = x_best[reset_mask]

                improvement.zero_()
                prev_eta = eta.clone()
                prev_f_best = f_best.clone()
                checkpoint_ptr += 1

            x_prev = x_k.clone()
            x_k = x_next.clone()

        # For random-start attacks, retain the clean sample if it has the
        # strongest CE objective among all retained candidates.
        if random_start:
            with torch.no_grad():
                f_clean = get_ce_loss_batch(model, x_clean, y, modelPlus)
                better_clean = f_clean > f_best
                x_best[better_clean] = x_clean[better_clean]

        x_adv_all[idx_out:idx_out + bs] = x_best.detach().cpu()
        y_all[idx_out:idx_out + bs] = y.detach().cpu()
        idx_out += bs

    return DMP.TensorToDataLoader(
        x_adv_all,
        y_all,
        transforms=None,
        batchSize=dataLoader.batch_size,
        randomizer=None,
    )

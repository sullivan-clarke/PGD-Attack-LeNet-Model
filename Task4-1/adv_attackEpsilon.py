import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor


# PGD Attack

def PGD(x, y, model, loss, niter=5, epsilon=0.3, stepsize=2/255,
        randinit=True, device="cpu"):
    """
    Projected Gradient Descent (PGD) adversarial attack (L-inf).

    Args:
        x         : clean input tensor  [N, C, H, W]
        y         : true labels         [N]
        model     : pre-trained model to attack
        loss      : loss function (e.g. nn.CrossEntropyLoss())
        niter     : number of PGD iterations
        epsilon   : L-inf perturbation budget (max pixel change)
        stepsize  : step size per iteration
        randinit  : if True, start from a random point inside the ε-ball
        device    : torch device string

    Returns:
        x_adv : adversarial examples  [N, C, H, W]  (detached, on CPU)
        y     : original labels       [N]            (on CPU)
    """
    model.eval()                          # no dropout / batchnorm randomness
    x, y = x.to(device), y.to(device)

    # ── 1. Initialise the perturbation ──────────────────────────────────────
    if randinit:
        # Uniform random start inside the ε-ball
        delta = torch.empty_like(x).uniform_(-epsilon, epsilon)
    else:
        delta = torch.zeros_like(x)

    # Ensure the starting point is already a valid image
    x_adv = torch.clamp(x + delta, 0.0, 1.0).detach()

    # ── 2. Iterative gradient ascent ────────────────────────────────────────
    for _ in range(niter):
        x_adv.requires_grad_(True)        # track gradients w.r.t. input

        logits = model(x_adv)
        cost   = loss(logits, y)          # scalar loss for the batch

        model.zero_grad()
        cost.backward()                   # ∂loss / ∂x_adv

        # print("max perturb:", (x_adv - x).abs().max().item())
        # print("grad mean:", x_adv.grad.abs().mean().item())
        
        with torch.no_grad():
            # FGSM step: move in the direction that *increases* the loss
            grad_sign = x_adv.grad.sign()
            x_adv = x_adv + stepsize * grad_sign
            # ── 3. Project back into the ε-ball around x ──────────────────
            # Clamp the total perturbation (x_adv - x) to [-ε, +ε]
            delta = torch.clamp(x_adv - x, -epsilon, epsilon)

            # Also keep pixel values in valid image range [0, 1]
            x_adv = torch.clamp(x + delta, 0.0, 1.0).detach()

    return x_adv.cpu(), y.cpu()


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    # Device
    device = (torch.accelerator.current_accelerator().type
              if torch.accelerator.is_available() else "cpu")
    print(f"Using {device} device")

    # Re-define the same LeNet used during training
    class LeNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 6, kernel_size=5),
                nn.ReLU(),
                nn.AvgPool2d(kernel_size=2, stride=2),
                nn.Conv2d(6, 16, kernel_size=5),
                nn.ReLU(),
                nn.AvgPool2d(kernel_size=2, stride=2),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(16 * 4 * 4, 120),
                nn.ReLU(),
                nn.Linear(120, 84),
                nn.ReLU(),
                nn.Linear(84, 10),
            )

        def forward(self, x):
            return self.classifier(self.features(x))

    # ── Load pre-trained weights ─────────────────────────────────────────────
    model = LeNet().to(device)
    model.load_state_dict(torch.load("Adam.pth", map_location=device))
    model.eval()
    print("Model loaded from Adam.pth")

    # ── MNIST test set (all 10 000 samples) ──────────────────────────────────
    test_data = datasets.MNIST(
        root="data", 
        train=False, 
        download=True, 
        transform=ToTensor()
    )

    test_loader = DataLoader(test_data, batch_size=256, shuffle=False)
    loss_fn = nn.CrossEntropyLoss()

    # ── Evaluate on CLEAN samples first (sanity check) ───────────────────────
    clean_correct = 0
    total         = 0

    print("\nEvaluating on clean test samples …")
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            preds = model(x_batch).argmax(dim=1)
            clean_correct += (preds == y_batch).sum().item()
            total         += y_batch.size(0)

    clean_acc = 100.0 * clean_correct / total
    print(f"  Clean accuracy : {clean_acc:.2f}%  ({clean_correct}/{total})")



    # PGD hyper-parameters (defaults from the spec)
    pgd_params = dict(
        niter    = 5,
        epsilon  = 0.3,
        stepsize = 2 / 255,
        randinit = True,
        device   = device,
    )

    print(f"\nCrafting adversarial examples with PGD …")
    print(f"  niter={pgd_params['niter']}, epsilon={pgd_params['epsilon']}, "
          f"stepsize={pgd_params['stepsize']:.5f}, randinit={pgd_params['randinit']}")

    with open("CSVs/Epsilon.csv", 'w') as f:
        pass
    with open("CSVs/Epsilon.csv", 'a') as f:
        f.write("Epsilon,Accuracy\n")
    
    epsilons = [0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0]
    for i in epsilons:
    # ── Evaluate on ADVERSARIAL samples ──────────────────────────────────────
        adv_correct = 0
        total       = 0
        processed = 0
        pgd_params = dict(
        niter    = 5,
        epsilon  = i,
        stepsize = 2 / 255,
        randinit = True,
        device   = device,
    )
        print(f"\nTRIAL {i}:")
        for batch_idx, (x_batch, y_batch) in enumerate(test_loader):
            # Craft adversarial examples for this batch
            x_adv, y_cpu = PGD(x_batch, y_batch, model, loss_fn, **pgd_params)

            # Evaluate the adversarial batch (no grad needed)
            with torch.no_grad():
                x_adv   = x_adv.to(device)
                y_cpu   = y_cpu.to(device)
                preds   = model(x_adv).argmax(dim=1)
                adv_correct += (preds == y_cpu).sum().item()
                total       += y_cpu.size(0)
                processed   += y_cpu.size(0)

            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(test_loader):
                print(f"  Processed {processed} / {len(test_data)} samples …",
                  file=sys.stderr)

        adv_acc = 100.0 * adv_correct / total
        with open("CSVs/Epsilon.csv", 'a') as f:
            f.write(f"{i},{adv_acc}\n")
        print(f"\n  Adversarial accuracy : {adv_acc:.2f}%  ({adv_correct}/{total})")
        print(f"\nSummary")
        print(f"  Clean accuracy      : {clean_acc:.2f}%")
        print(f"  Adversarial accuracy: {adv_acc:.2f}%")
        print(f"  Accuracy drop       : {clean_acc - adv_acc:.2f} pp")
def calculateLoss(reward: float, target: float) -> float:
    return pow(reward-target, 2)


def isWithinLossThreshold(loss: float, threshold: float) -> bool:
    return loss < threshold


def normaliseLossDistr(lossDistr: list[float]) -> list[float]:
    normalisedDistr: list[float] = []
    min_val: float = min(lossDistr)
    max_val: float = max(lossDistr)
    for n in lossDistr:
        normalisedDistr.append((n-min_val)/(max_val-min_val))
    return normalisedDistr


def findMeanSquaredError(lossDistr: list[float]) -> float:
    return sum(lossDistr) / len(lossDistr)

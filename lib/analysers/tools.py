def calculateLoss(reward: float, target: float) -> float:
    return pow(reward-target, 2)


def isWithinLossThreshold(loss: float, threshold: float) -> bool:
    return loss < threshold


def normaliseLossDistr(lossDistr: list[float]) -> list[float]:
    normalisedDistr: list[float] = []
    min: float = lossDistr[0]
    max: float = lossDistr[-1]
    for n in lossDistr:
        normalisedDistr.append((n-min)/(max-min))
    return normalisedDistr

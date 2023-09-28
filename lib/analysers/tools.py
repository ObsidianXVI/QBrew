import scipy.integrate as integrate


def calculateLoss(reward: float, target: float) -> float:
    return pow(reward-target, 2)


def isWithinLossThreshold(loss: float, threshold: float) -> bool:
    return loss < threshold


def normaliseDistr(lossDistr: list[float]) -> list[float]:
    normalisedDistr: list[float] = []
    min_val: float = min(lossDistr)
    max_val: float = max(lossDistr)
    for n in lossDistr:
        normalisedDistr.append((n-min_val)/(max_val-min_val))
    return normalisedDistr


def findMeanSquaredError(lossDistr: list[float]) -> float:
    return sum(lossDistr) / len(lossDistr)


def findReconvergenceTime(rewDistr: list[float], target: float, threshold: int, timeThreshold: int):
    lower: float = target - threshold
    upper: float = target + threshold
    stack: list[float] = []
    timesteps: int = 0
    for n in rewDistr:
        timesteps += 1
        if (lower <= n <= upper):
            stack.append(n)
        else:
            stack.clear()

        if (len(stack) == timeThreshold):
            return timesteps
    return None


def sortLossDistr(lossDistr: list[float]) -> list[float]:
    return sorted(lossDistr, reverse=True)


def computeValueAtRisk(lossDistr: list[float], timestepDistr: list[float], confidence: int) -> tuple[float, int]:
    interval: int = 100 - confidence
    index: int = -1
    for t in timestepDistr:
        index += 1
        if (round(t * 100) == interval):
            break

    return lossDistr[index], index


def computeConditionalVAR(lossDistr: list[float], timestepDistr: list[float], confidence: int) -> float:
    confidenceDec: float = confidence / 100
    integral: float = integrate.quad(lambda x: computeValueAtRisk(
        lossDistr, timestepDistr, int(x * 100))[0], confidenceDec, 1)
    return (1/(1-confidenceDec)) * float(integral[0])

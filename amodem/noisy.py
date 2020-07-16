import numpy as np


def gimmeX(x):
    return x


class Noisy:
    """creates a noise maker function"""

    _aggFunc = gimmeX

    def __init__(self, aggFunc=None):
        self.aggFunc = aggFunc or self._aggFunc

    # pylint: disable=too-many-arguments
    def shotIndices(self, shotL=4.0, start=0, cnt=-1, groupSpan=-1,
                    aggFunc=None):
        """Makes Lists of indices for injecting shot noise into signal """
        shotbatch = 20
        myaggFunc = aggFunc or self.aggFunc
        idx = []
        span = []
        thisGroup, lastGroup, firstGroupStart = (0, 0, start)
        lastFill = start
        while cnt != 0:
            if idx:
                cnt -= 1
                span += [idx.pop()]
                if groupSpan > 0:
                    # span group starts based on first entry in the span
                    thisGroup = (span[0] - firstGroupStart) // groupSpan
                    while lastGroup < thisGroup - 1:
                        lastGroup += 1
                        yield []
                    lastGroup = thisGroup
                    if (span[-1] - firstGroupStart) // groupSpan > thisGroup:
                        yield [myaggFunc(i) for i in span[:-1]]
                        span = [span[-1]]
                else:
                    yield myaggFunc(span.pop())
            else:
                idx = list(
                    np.cumsum(np.random.poisson(lam=shotL, size=shotbatch))
                    + lastFill)
                idx.reverse()
                lastFill = idx[0]
        if span:
            if groupSpan > 0:
                yield [myaggFunc(i) for i in span]
            else:
                yield myaggFunc(span[0])
        return 0

    def noisemaker(
            self, norm_stdev=1.0, shot_stdev=1.0, shotlambda=-1.0,
            batchSize=32, count=-1, aggFunc=None):
        """Creates AWGN with shot noise(optionally)"""

        if shotlambda > 0:
            if aggFunc:
                shotGen = self.shotIndices(
                    shotL=shotlambda, groupSpan=batchSize, aggFunc=aggFunc)
            else:
                shotGen = self.shotIndices(
                    shotL=shotlambda, groupSpan=batchSize,
                    aggFunc=lambda x: x % batchSize)
        else:
            shotGen = None
        while count != 0:
            count -= 1
            span = np.random.normal(scale=norm_stdev, size=batchSize)
            # yield span
            if shotlambda > 0:
                for ii in next(shotGen, []) if shotGen else []:
                    span[ii] += np.random.normal(scale=shot_stdev)
            yield span
        return count


if __name__ == "__main__":
    # pylint: disable=import-error,import-outside-toplevel
    import pylab as plt
    n = Noisy()
    fig2 = plt.figure(figsize=(10, 6))
    ssss = n.noisemaker(
        norm_stdev=0.125, shot_stdev=1.0, shotlambda=20.0,
        batchSize=512, count=5)
    for i, sss in enumerate(ssss):
        plt.plot(
            sss + 10 * (i), "-", label="Slice {}".format(i),
        )
    plt.legend()
    plt.show()

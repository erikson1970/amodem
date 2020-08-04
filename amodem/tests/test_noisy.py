from amodem import noisy


def test_awgn():
    n = noisy.Noisy()
    for ns in [0.5 ** i for i in range(5)]:
        ssss = n.noisemaker(norm_stdev=ns, shotlambda=-1.0,
                            batchSize=2 ** 16, count=30)
        assert abs(sum([1.0 - noisy.np.std(i) / ns for i in ssss])) < 0.1


def test_shot():
    n = noisy.Noisy()
    for ns in [2.0 ** i for i in range(1, 10)]:
        ssss = n.noisemaker(norm_stdev=0.0, shot_stdev=1.0, shotlambda=ns,
                            batchSize=2 ** 14, count=15)
        assert abs(1.0 - (sum([noisy.np.std(i) for i in ssss]) / 15.0)
                   * noisy.np.sqrt(ns)) < 0.1

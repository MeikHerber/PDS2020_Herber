def sample_fraction(fraction, x, y):
    if fraction == 1:
        x['y'] = y
        x = x.drop('y', axis=1)
        return x, y
    else:
        x['y'] = y
        sample = x.sample(frac=fraction)
        x_sample = sample.drop('y', axis=1)
        y_sample = sample['y']
        return x_sample, y_sample

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from helper.plot import plot_image
from helper.preprocess import get_inputs
from feature.color import ColorFeature
from feature.histogram import HistogramFeature
from feature.blob import BlobFeature
from feature.gradient import GradientFeature
from feature.geo import GeoFeature
from feature.extension import ExtensionFeature
from feature.size import SizeFeature
from feature.words import WordsFeature


def get_extractors(visual=True, textual=True):
    extractors = []
    if visual:
        extractors += [
            ColorFeature,
            HistogramFeature,
            BlobFeature,
            GradientFeature
        ]
    if textual:
        extractors += [
            GeoFeature,
            ExtensionFeature,
            SizeFeature,
            WordsFeature
        ]
    return extractors

def feature_vector(filename):
    inputs = get_inputs(filename)
    extractors = get_extractors()
    features = apply_extractors(inputs, extractors)
    return list(features)

def feature_names():
    names = []
    for extractor in get_extractors():
        for name in extractor.names():
            names.append(name)
    return names

def apply_extractors(inputs, extractors):
    for extractor in extractors:
        yield from extractor(**inputs).extract()

def print_features(names, features):
    print(len(names), 'features extracted:')
    for name, feature in zip(names, features):
        print('{name: <25} {feature: >8.4f}'.format(**locals()))


if __name__ == '__main__':
    parser = ArgumentParser(description='Extract feature vector of an image.',
        formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('filename',
        help='Path to the image to extract features from')
    args = parser.parse_args()

    # inputs = get_inputs(args.filename)
    # plot_image(inputs['original'])
    # plot_image(inputs['image'])
    # BlobFeature(**inputs).show()
    # GradientFeature(**inputs).show()

    names = feature_names()
    features = feature_vector(args.filename)
    print_features(names, features)

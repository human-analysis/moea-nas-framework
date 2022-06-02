from ofa.utils.layers import LinearLayer


def reset_classifier(model, n_classes, dropout_rate=0.0):
    last_channel = model.classifier.in_features
    model.classifier = LinearLayer(last_channel, n_classes, dropout_rate=dropout_rate)
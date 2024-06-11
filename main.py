from Layer import Layer

LAYER1 = 785
LAYER2 = 16
LAYER3 = 10


def main():
    layer1 = Layer(n_count=LAYER1, prev_count=0)

    layer2 = Layer(n_count=LAYER2, prev_count=LAYER1)
    layer1.add_layer(layer2)

    layer3 = Layer(n_count=LAYER3, prev_count=LAYER2)
    layer1.add_layer(layer3)

    layer1.save('test')
    loaded_layer = Layer.load('test')
    print(f'original:\n{layer1}\n')
    # print(f"load:\n{loaded_layer}")
    # print(layer1.weights.shape)
    # print(loaded_layer == layer1)


    test = Layer.load_dataset('data//mnist_test.csv')
    # train = Layer.load_dataset('data//mnist_train.csv')
    print(test[0][0])
    res = layer1.process(test[0][1])
    print(res)


if __name__ == '__main__':
    main()

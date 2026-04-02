import matplotlib.pyplot as plt

def data_imbalance(data):
    plt.figure(figsize=(5, 5))
    plt.hist(data['Accident_severity'], color='blue', bins=5)
    plt.title('Accident severity distribution')
    plt.xlabel('Accident severity')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

def plot_2d(X_red, y, title):
    plt.figure(figsize=(8,8))

    colors = {
        0: 'blue',
        1: 'green',
        2: 'white'
    }

    markers = {
        0: 'o',   # circle
        1: 's',   # square
        2: '^'    # triangle
    }

    for cls in sorted(set(y)):
        plt.scatter(
            X_red[y == cls, 0],
            X_red[y == cls, 1],
            label=f"Class {cls}",
            alpha=0.7,
            color=colors.get(cls, 'black'),
            marker=markers.get(cls, 'o'),
            s=60   # size of points
        )

    plt.legend()
    plt.title(title)
    plt.show()
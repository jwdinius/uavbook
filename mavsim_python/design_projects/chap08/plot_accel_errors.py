import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    errors_x, errors_y, errors_z = [], [], []
    with open("accel_errors.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            x, y, z = [float(val) for val in line.split(',')]
            errors_x.append(x)
            errors_y.append(y)
            errors_z.append(z)
    
    plt.figure()
    x = [_x for _x in range(len(errors_x))]
    std_x = np.std(errors_x)
    std_y = np.std(errors_y)
    std_z = np.std(errors_z)
    plt.plot(x, errors_x, 'b', label="ax")
    plt.plot(x, errors_y, 'r', label="ay")
    plt.plot(x, errors_z, 'g', label="az")
    plt.plot([x[0], x[-1]], [3*std_x, 3*std_x], 'b--', [x[0], x[-1]], [-3*std_x, -3*std_x], 'b--')
    plt.plot([x[0], x[-1]], [3*std_y, 3*std_y], 'r--', [x[0], x[-1]], [-3*std_y, -3*std_y], 'r--')
    plt.plot([x[0], x[-1]], [3*std_z, 3*std_z], 'g--', [x[0], x[-1]], [-3*std_z, -3*std_z], 'g--')
    plt.ylim((-4, 4))
    plt.legend()
    plt.xlabel("Timestep")
    plt.ylabel("Model error (3-sigma)")
    plt.title("Model Error Introduced Per Channel")
    print(f"1-sigma errors: {std_x}, {std_y}, {std_z}")
    plt.show()

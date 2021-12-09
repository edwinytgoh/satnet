import os

package_path = os.path.dirname(os.path.abspath(__file__))
path = os.path.dirname(package_path)
data_path = os.path.join(path, "data")

problems = {
    2018: os.path.join(data_path, "problems.json"),
}

maintenance = {2018: os.path.join(data_path, "maintenance.csv")}

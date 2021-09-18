from ada_test import main_cmim as main_cmim_ada
from ada_test import main_std_cmim as main_std_cmim_ada
from bagging_test import main_cmim as main_cmim_bag
from bagging_test import main_std_cmim as main_std_cmim_bag
from rf_test import main_cmim as main_cmim_rf
from rf_test import main_std_cmim as main_std_cmim_rf


def main():
    n_jobs = 5
    for func in (main_cmim_rf, main_std_cmim_rf, main_cmim_ada,
                 main_std_cmim_ada, main_cmim_bag, main_std_cmim_bag):
        for n_cmim in (2, 4):
            func(n_cmim, True, n_jobs=n_jobs)


if __name__ == '__main__':
    main()

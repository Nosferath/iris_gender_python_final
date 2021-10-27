# 480x80 TESTS ARE CURRENTLY COMMENTED ON constants.py
def perform_regular_tests(n_jobs: int):
    from ada_test import main as main_ada
    from ada_test import main_std as main_std_ada
    from bagging_test import main as main_bag
    from bagging_test import main_std as main_std_bag
    from rf_test import main as main_rf
    from rf_test import main_std as main_std_rf
    tests = [main_ada, main_std_ada, main_bag, main_std_bag, main_rf,
             main_std_rf]
    for t in tests:
        t(n_jobs=n_jobs)


def perform_regular_cmim_tests(n_jobs: int):
    from ada_test import main_cmim as main_ada
    from ada_test import main_std_cmim as main_std_ada
    from bagging_test import main_cmim as main_bag
    from bagging_test import main_std_cmim as main_std_bag
    from rf_test import main_cmim as main_rf
    from rf_test import main_std_cmim as main_std_rf
    tests = [main_ada, main_std_ada, main_bag, main_std_bag, main_rf,
             main_std_rf]
    for t in tests:
        for n_cmim in (2, 4):
            t(n_cmim=n_cmim, n_jobs=n_jobs)


def perform_mod_v2_cmim_tests(n_jobs: int):
    from ada_test import main_cmim_mod_v2 as main_ada
    from ada_test import main_std_cmim_mod_v2 as main_std_ada
    from bagging_test import main_cmim_mod_v2 as main_bag
    from bagging_test import main_std_cmim_mod_v2 as main_std_bag
    from rf_test import main_cmim_mod_v2 as main_rf
    from rf_test import main_std_cmim_mod_v2 as main_std_rf
    tests = [main_ada, main_std_ada, main_bag, main_std_bag, main_rf,
             main_std_rf]
    for t in tests:
        for n_cmim in (2, 4):
            t(n_cmim=n_cmim, n_jobs=n_jobs)

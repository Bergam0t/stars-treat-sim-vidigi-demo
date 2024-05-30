"""Model testing
 
This module provides a set of tests to run against `treat_sim`.  
These tests are either pass or fail and no interpretation is needed. 
 
Model testing is work in progress and more tests will be 
added in due course as we refine and/or expand `treat_sim` functionality.
 
We have broken the testing of `treat_sim` into the following sections
 
1. Functionality tests:.  
These provide tests of the overall setup and features 
of using the model including the ability to do repeatable 
single and multiple replications.

2. Extreme value tests: 
The model is configured with extreme values for its parameters. 
For example, limiting arrivals, blocking routes, infinite capacity.  
This helps test that the logic of the model is implemented correctly.

3. Deterministic run tests:*
Test if the total time in the pathway output behaves as 
expected if activites are converted to a deterministic 
values and capacity is removed. 

4. Dirty tests**: TO-DO
tests to check that `treat-sim` fails as 
expected when given certain values.
"""

from treat_sim.model import (
    Scenario,
    TreatmentCentreModel,
    single_run,
    multiple_replications,
    DEFAULT_N_TRIAGE,
    DEFAULT_N_REG,
    DEFAULT_REG_MEAN,
    DEFAULT_TRAUMA_TREAT_MEAN,
    DEFAULT_PROB_TRAUMA,
    Exponential,
    Uniform,
)

import pandas as pd
import pytest

# types library is used to overwrite methods from treat_sim
import types

# sim_tools fixed dist is used for deterministic runs
from sim_tools.distributions import FixedDistribution


from pathlib import Path

ARRIVAL_FILE_1 = "ed_arrivals.csv"
ARRIVAL_FILE_2 = "ed_arrivals_test_1.csv"
PATH_ARRIVAL_1 = Path(__file__).parent.joinpath("data", ARRIVAL_FILE_1)
PATH_ARRIVAL_2 = Path(__file__).parent.joinpath("data", ARRIVAL_FILE_2)

# Tests
#
# 1 Functionality tests
#
# Here we test that various modes of running the model work correctly.
# These include
#
# * single run mode
# * multiple replications mode
# * repeatable results using random number sets.
# * results collection period
# * the expected number of patient arrivals given an arrival profile
# * experimental input parameters remain constant (unchanged) through run
#


def test_single_run_results_length_and_type():
    """
    Test a single_run of the model.

    The single_run function should return a pandas.DataFrame
    containing 16 columns and a single row.

     0   00_arrivals
     1   01a_triage_wait
     2   01b_triage_util
     3   02a_registration_wait
     4   02b_registration_util
     5   03a_examination_wait
     6   03b_examination_util
     7   04a_treatment_wait(non_trauma)
     8   04b_treatment_util(non_trauma)
     9   05_total_time(non-trauma)
     10  06a_trauma_wait
     11  06b_trauma_util
     12  07a_treatment_wait(trauma)
     13  07b_treatment_util(trauma)
     14  08_total_time(trauma)
     15  09_throughput

    Expected result:
    ---------------
        len(run_results) == 16 and isinstance(run_results, pd.Dataframe)

    Returns:
    -------
    bool: does the model pass the test.
    """
    EXPECTED_LENGTH = 16

    # a default experiment
    default_experiment_params = Scenario()

    # run the model in single run model
    run_results = single_run(default_experiment_params, random_no_set=41)

    # test
    assert len(run_results.T) == EXPECTED_LENGTH and isinstance(
        run_results, pd.DataFrame
    )


@pytest.mark.parametrize("n_reps", [(1), (2), (10), (23)])
def test_multiple_replications_results_length_and_type(n_reps):
    """
    Test running the model in multiple replications mode

    The multiple function should return a pandas.DataFrame
    containing 16 columns and n_reps rows. Its shape is (n_reps, 16)

     0   00_arrivals
     1   01a_triage_wait
     2   01b_triage_util
     3   02a_registration_wait
     4   02b_registration_util
     5   03a_examination_wait
     6   03b_examination_util
     7   04a_treatment_wait(non_trauma)
     8   04b_treatment_util(non_trauma)
     9   05_total_time(non-trauma)
     10  06a_trauma_wait
     11  06b_trauma_util
     12  07a_treatment_wait(trauma)
     13  07b_treatment_util(trauma)
     14  08_total_time(trauma)
     15  09_throughput

    Expected result:
    ---------------
        rep_results.shape == (n_reps, 16) and isinstance(rep_results, pd.DataFrame)
    """
    EXPECTED_N_COLUMNS = 16
    EXPECTED_SHAPE = (n_reps, EXPECTED_N_COLUMNS)

    # a default experiment
    default_experiment_params = Scenario()

    # run the model in multiple replications mode
    rep_results = multiple_replications(default_experiment_params, n_reps=n_reps)

    # test
    assert rep_results.shape == EXPECTED_SHAPE and isinstance(rep_results, pd.DataFrame)


@pytest.mark.parametrize(
    "random_number_set",
    [
        (0),
        (1),
        (2),
        (101),
        (42),
    ],
)
def test_random_number_set(random_number_set):
    """
    Test the model produces repeatable results
    given the same set set of random seeds.

    Expected result:
    ---------------
        difference between data frames is 0.0
    """

    results = []

    for i in range(2):

        exp = Scenario()

        # run the model in single run model
        run_results = single_run(exp, random_no_set=random_number_set)

        results.append(run_results)

    # test
    assert (results[0] - results[1]).sum().sum() == 0.0


@pytest.mark.parametrize(
    "rc_period",
    [
        (10.0),
        (1_000.0),
        (25.0),
        (500.0),
        (143.0),
    ],
)
def test_run_length_control(rc_period):
    """
    Test that the simulation model can be set up
    to run different results collection period.

    """
    scenario = Scenario()

    # set random number set - this controls sampling for the run.
    scenario.set_random_no_set(42)

    # create an instance of the model
    model = TreatmentCentreModel(scenario)

    # run the model
    model.run(results_collection_period=rc_period)

    # run results
    assert model.env.now == rc_period


@pytest.mark.parametrize(
    "arrival_profile",
    [
        (PATH_ARRIVAL_1),
        (PATH_ARRIVAL_2),
    ],
)
def test_arrival_numbers(arrival_profile):
    """
    Test that the number of arrivals in a day is correct
    on average using a different arrival profiles.
    This includes the default used in the application.

    The multiple_replications function is used to generate
    the mean number of arrivals over 100 replications.

    We allow for a 1% sampling deviation.

    00_arrivals

    Expected result:
    ---------------
        mean arrivals = sum(arrival_rate from input file)

    """

    # this function replaces Scenario.init_nspp()
    # We use this approach because in treat_sim v2.0.0
    # the path to the profile is hard coded and cannot be
    # set directly when the Scenario is created.
    def init_nspp_for_testing(self):

        # read arrival profile
        self.arrivals = pd.read_csv(arrival_profile)
        self.arrivals["mean_iat"] = 60 / self.arrivals["arrival_rate"]
        print(sum(self.arrivals["arrival_rate"]))

        # maximum arrival rate (smallest time between arrivals)
        self.lambda_max = self.arrivals["arrival_rate"].max()

        # thinning exponential
        self.arrival_dist = Exponential(
            60.0 / self.lambda_max, random_seed=self.seeds[8]
        )

        # thinning uniform rng
        self.thinning_rng = Uniform(low=0.0, high=1.0, random_seed=self.seeds[9])

    arrival_data = pd.read_csv(arrival_profile)
    expected_patients = arrival_data["arrival_rate"].sum()

    # a default experiment
    scenario = Scenario()
    # overwrite the nspp init function in the scenario class
    scenario.init_nspp = types.MethodType(init_nspp_for_testing, scenario)

    # run the model in multiple_reps mode
    results = multiple_replications(scenario, n_reps=100)
    # summary results...
    mean_results = results.mean().round(2)

    # test
    assert pytest.approx(mean_results["00_arrivals"], rel=0.01) == expected_patients


# In[10]:


@pytest.mark.parametrize(
    "n_triage, n_reg, reg_mean, trauma_treat_mean, prob_trauma",
    [
        (
            DEFAULT_N_TRIAGE,
            DEFAULT_N_REG,
            DEFAULT_REG_MEAN,
            DEFAULT_TRAUMA_TREAT_MEAN,
            DEFAULT_PROB_TRAUMA,
        ),
        (
            5,
            DEFAULT_N_REG,
            DEFAULT_REG_MEAN,
            DEFAULT_TRAUMA_TREAT_MEAN,
            DEFAULT_PROB_TRAUMA,
        ),
        (DEFAULT_N_TRIAGE, 5, DEFAULT_REG_MEAN, DEFAULT_TRAUMA_TREAT_MEAN, 0.5),
        (DEFAULT_N_TRIAGE, DEFAULT_N_REG, 15.0, DEFAULT_TRAUMA_TREAT_MEAN, 0.25),
        (42, DEFAULT_N_REG, DEFAULT_REG_MEAN, 65.2, DEFAULT_PROB_TRAUMA),
    ],
)
def test_parameters_stay_constant(
    n_triage, n_reg, reg_mean, trauma_treat_mean, prob_trauma
):
    """
    Test that the Scenario parameters are not
    modified during the run.

    Recommended by Banks et al in verification section.

    We modify the test slightly by running multiple
    tests with a small selection of parameters.

    Expected result:
    ---------------
    No change to the experimental parameters during a run.
    """

    def scenario_params_are_equal(to_test, before_run):

        return (
            to_test.n_triage == before_run.n_triage
            and to_test.n_reg == before_run.n_reg
            and to_test.n_exam == before_run.n_exam
            and to_test.n_trauma == before_run.n_trauma
            and to_test.n_cubicles_1 == before_run.n_cubicles_1
            and to_test.n_cubicles_2 == before_run.n_cubicles_2
            and to_test.triage_mean == before_run.triage_mean
            and to_test.reg_mean == before_run.reg_mean
            and to_test.reg_var == before_run.reg_var
            and to_test.exam_mean == before_run.exam_mean
            and to_test.exam_var == before_run.exam_var
            and to_test.exam_min == before_run.exam_min
            and to_test.trauma_mean == before_run.trauma_mean
            and to_test.trauma_treat_mean == before_run.trauma_treat_mean
            and to_test.trauma_treat_var == before_run.trauma_treat_var
            and to_test.non_trauma_treat_mean == before_run.non_trauma_treat_mean
            and to_test.non_trauma_treat_var == before_run.non_trauma_treat_var
            and to_test.non_trauma_treat_p == before_run.non_trauma_treat_p
            and to_test.prob_trauma == before_run.prob_trauma
        )

    # this is the scenario we run
    scenario_to_run_with_model = Scenario(
        n_triage=n_triage,
        n_reg=n_reg,
        reg_mean=reg_mean,
        trauma_treat_mean=trauma_treat_mean,
        prob_trauma=prob_trauma,
    )

    # this is a copy of the scenario that we use to test that there is no modification to parmas
    scenario_to_compare = Scenario(
        n_triage=n_triage,
        n_reg=n_reg,
        reg_mean=reg_mean,
        trauma_treat_mean=trauma_treat_mean,
        prob_trauma=prob_trauma,
    )

    # run model - we are not interested in restuls.
    _ = single_run(scenario_to_run_with_model, random_no_set=101)

    # compare
    assert scenario_params_are_equal(scenario_to_run_with_model, scenario_to_compare)


# ### 3.2. Extreme value tests
#
# Here we manipulate the input parameters of the model to test that it behaves as expected. We run the following tests
#
# * All arrivals are trauma
# * All arrivals are non-trauma
# * Infinite capacity for activities
# * All non-trauma patients require treatment
# * All non-trauma patients do not require treatment
# * Block trauma arrivals at stabilisation
# * Block non-trauma arrivals at examination
# * Block all arrivals at triage

# In[11]:


### NOTE: we are ignoring mean of empty array warnings from numpy using filterwarnings
### We will handle this in a future release of treat_sim.


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize(
    "random_no_set",
    [
        (42),
        (1),
        (754),
        (9876534321),
        (76546783986555),
    ],
)
def test_all_trauma(random_no_set):
    """
    Force all patients to use trauma pathway

    i.e. Set prob_trauma = 1.0

    Expected result:
    -------
    No patients use the non-trauma pathway this means that a number of KPIs
    are NaN (not a number). These include:
    '02a_registration_wait'
    '05_total_time(non-trauma)'

    Trauma pathway KPIs are valid numbers. We test total time in
    system: '08_total_time(trauma)'

    """

    # create a new scenario and set prob of trauma to 100%
    scenario = Scenario(prob_trauma=1.0)

    # run the model in single run model
    run_results = single_run(scenario, random_no_set=random_no_set)

    # run results
    assert (
        pd.isna(run_results["05_total_time(non-trauma)"].iloc[0])
        and pd.isna(run_results["02a_registration_wait"].iloc[0])
        and not pd.isna(run_results["08_total_time(trauma)"].iloc[0])
    )


# In[12]:


### NOTE: we are ignoring mean of empty array warnings from numpy using filterwarnings
### We will handle this in a future release of treat_sim.


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize(
    "random_no_set",
    [
        (42),
        (1),
        (754),
        (9876534321),
        (76546783986555),
    ],
)
def test_all_nontrauma(random_no_set):
    """
    Force all patients to use the non-trauma pathway

    i.e. Set prob_trauma = 0.0

    Expected result:
    -------
    No patients use the trauma pathway this means that a number of KPIs
    if NaN (Not a Number) as no patient use the activities.

    '06a_trauma_wait'
    '07a_treatment_wait(trauma)'
    '08_total_time(trauma)'

    Non-trauma pathway KPIs are valid numbers. We test
    '02a_registration_wait'
    '05_total_time(non-trauma)'
    """

    # create a new scenario and set prob of trauma to 100%
    scenario = Scenario(prob_trauma=0.0)

    # run the model in single run model
    run_results = single_run(scenario, random_no_set=random_no_set)

    # run results
    assert (
        pd.isna(run_results["06a_trauma_wait"].iloc[0])
        and pd.isna(run_results["07a_treatment_wait(trauma)"].iloc[0])
        and pd.isna(run_results["08_total_time(trauma)"].iloc[0])
        and not pd.isna(run_results["02a_registration_wait"].iloc[0])
        and not pd.isna(run_results["05_total_time(non-trauma)"].iloc[0])
    )


# In[13]:


@pytest.mark.parametrize(
    "random_no_set",
    [(42), (1), (754), (9876534321), (76546783986555), (9876888854637815463789)],
)
def test_infinite_capacity(random_no_set):
    """
    Remove all capacity constraints in the model
    by setting all capacity to M a large number

    Expected result:
    -------
    No queuing. The following KPIs = 0.0

    01a_triage_wait
    02a_registration_wait
    03a_examination_wait
    04a_treatment_wait(non_trauma)
    06a_trauma_wait'
    07a_treatment_wait(trauma)
    """

    M = 100_000_000

    # create a new scenario and set prob of trauma to 100%
    scenario = Scenario(
        n_triage=M, n_reg=M, n_exam=M, n_trauma=M, n_cubicles_1=M, n_cubicles_2=M
    )

    # run the model in single run model
    run_results = single_run(scenario, random_no_set=random_no_set)

    assert (
        run_results["01a_triage_wait"].iloc[0] == 0.0
        and run_results["02a_registration_wait"].iloc[0] == 0.0
        and run_results["03a_examination_wait"].iloc[0] == 0.0
        and run_results["04a_treatment_wait(non_trauma)"].iloc[0] == 0.0
        and run_results["06a_trauma_wait"].iloc[0] == 0.0
        and run_results["07a_treatment_wait(trauma)"].iloc[0] == 0.0
    )


# In[14]:


### NOTE: we are ignoring mean of empty array warnings from numpy using filterwarnings
### We will handle this in a future release of treat_sim.


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize(
    "random_no_set",
    [
        (42),
        (1),
        (754),
        (9876534321),
        (76546783986555),
    ],
)
def test_no_treatment_for_nontrauma(random_no_set):
    """
    Force all non-trauma patients to be discharged without treatment

    i.e. non_trauma_treat_p = 0.0

    Expected result:
    -------
    No non-trauma patients queue or use treatment cubicles
    The following variable is NaN

    04a_treatment_wait(non_trauma)

    and the utilisation of the non-trauma cubicles is 0

    04b_treatment_util(non_trauma)

    """
    # create a new scenario and set prob of trauma to 100%
    scenario = Scenario(non_trauma_treat_p=0.0)

    # run the model in single run model
    run_results = single_run(scenario, random_no_set=random_no_set)

    # run results
    assert (
        pd.isna(run_results["04a_treatment_wait(non_trauma)"].iloc[0])
        and run_results["04b_treatment_util(non_trauma)"].iloc[0] == 0.0
    )


# In[15]:


### NOTE: we are ignoring mean of empty array warnings from numpy using filterwarnings
### We will handle this in a future release of treat_sim.


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize(
    "random_no_set",
    [
        (42),
        (1),
        (754),
        (9876534321),
        (76546783986555),
    ],
)
def test_all_nontrauma_no_treatment(random_no_set):
    """
    Force all patients to use the non-trauma pathway

    i.e. Set prob_trauma = 0.0 and non_trauma_treat_p = 0.0

    Expected result:
    -------
    No patients use the trauma pathway this means that a number of KPIs
    if NaN (Not a Number) as no patient use the activities.

    '06a_trauma_wait'
    '07a_treatment_wait(trauma)'
    '08_total_time(trauma)'

    Non-trauma pathway KPIs are valid numbers. We test
    '02a_registration_wait'
    '05_total_time(non-trauma)'

    No non-trauma patients queue or use treatment cubicles
    The following variable is NaN

    04a_treatment_wait(non_trauma)

    and the utilisation of the non-trauma cubicles is 0

    04b_treatment_util(non_trauma)

    """

    # create a new scenario and set prob of trauma to 100%
    scenario = Scenario(prob_trauma=0.0, non_trauma_treat_p=0.0)

    # run the model in single run model
    run_results = single_run(scenario, random_no_set=random_no_set)

    # run results
    assert (
        pd.isna(run_results["06a_trauma_wait"].iloc[0])
        and pd.isna(run_results["07a_treatment_wait(trauma)"].iloc[0])
        and pd.isna(run_results["08_total_time(trauma)"].iloc[0])
        and not pd.isna(run_results["02a_registration_wait"].iloc[0])
        and not pd.isna(run_results["05_total_time(non-trauma)"].iloc[0])
        and pd.isna(run_results["04a_treatment_wait(non_trauma)"].iloc[0])
        and run_results["04b_treatment_util(non_trauma)"].iloc[0] == 0.0
    )


### NOTE: we are ignoring mean of empty array warnings
## from numpy using filterwarnings
### We will handle this in a future release of treat_sim.


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize(
    "random_no_set",
    [
        (42),
        (1),
        (754),
        (9876534321),
        (76546783986555),
    ],
)
def test_block_trauma_at_stabilisation(random_no_set):
    """
    Block all trauma patients at stablisation
    This is achieved by setting mean trauma time to
    M a very large number.

    Expected result:
    -------
    Trauma pathway total_time in system = NaN
    Non_trama pathway operates as expected. Not NaN
    """
    M = 100_000_000

    # create a new scenario and set prob of trauma to 100%
    scenario = Scenario(trauma_mean=M)

    # run the model in single run model
    run_results = single_run(scenario, random_no_set=random_no_set)

    # run results
    assert pd.isna(run_results["08_total_time(trauma)"].iloc[0]) and not pd.isna(
        run_results["05_total_time(non-trauma)"].iloc[0]
    )


### NOTE: we are ignoring mean of empty array warnings from numpy using filterwarnings
### We will handle this in a future release of treat_sim.


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize(
    "random_no_set",
    [
        (42),
        (1),
        (754),
        (9876534321),
        (76546783986555),
    ],
)
def test_block_nontrauma_at_examination(random_no_set):
    """
    Block all non trauma patients at examination
    This is achieved by setting mean examination time to
    M a very large number.

    Expected result:
    -------
    Trauma pathway total_time in system is normal i.e. not a NaN
    Non_trama pathway total pathway time is NaN and waiting time for
    treatment is NaN
    """
    M = 100_000_000

    # create a new scenario and set prob of trauma to 100%
    scenario = Scenario(exam_mean=M)

    # run the model in single run model
    run_results = single_run(scenario, random_no_set=random_no_set)

    # run results
    assert (
        not pd.isna(run_results["08_total_time(trauma)"].iloc[0])
        and pd.isna(run_results["04a_treatment_wait(non_trauma)"].iloc[0])
        and pd.isna(run_results["05_total_time(non-trauma)"].iloc[0])
    )


### NOTE: we are ignoring mean of empty array warnings from numpy using filterwarnings
### We will handle this in a future release of treat_sim.
@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize(
    "random_no_set",
    [
        (42),
        (1),
        (754),
        (9876534321),
        (76546783986555),
    ],
)
def test_block_all_arrivals_at_triage(random_no_set):
    """
    Block all all patient arrivals at triage
    This is achieved by setting mean triage time to
    M a very large number.  We will set triage capacity to 1.
    This also means that the queue length of triage will be arrivals - 1

    Expected result:
    -------
    Trauma pathway total_time in system is normal i.e. not a NaN
    Non_trama pathway total pathway time is NaN and waiting time for
    treatment is NaN

    Notes:
    -----
    After a TreatmentCentreModel has been initialised by single_run
    a variable called `triage` is available. This is the simpy.Resource
    representing triage capacity.

    """
    M = 100_000_000

    # create a new scenario and set prob of trauma to 100%
    scenario = Scenario(triage_mean=M, n_triage=1)

    # run the model in single run model
    run_results = single_run(scenario, random_no_set=random_no_set)

    # run results
    assert (
        pd.isna(run_results["08_total_time(trauma)"].iloc[0])
        and pd.isna(run_results["05_total_time(non-trauma)"].iloc[0])
        and len(scenario.triage.queue) == run_results["00_arrivals"].iloc[0] - 1
    )


# 3. Deterministic activities
# We have simplifed the model to a deterministic run by replacing
# all activity distributions with a fixed static value.


def init_sampling_to_fixed(self):
    """
    Create the distributions used by the model and initialise
    the random seeds of each.  This modified function
    makes use of a fixed distribution where the fixed value
    is the mean of the original sampling distribution.

    We ignore arrivals as this is handled by init_nspp()

    """
    # create distributions - all are fixed apart from arrivals

    # Triage duration
    self.triage_dist = FixedDistribution(self.triage_mean)

    # Registration duration (non-trauma only)
    self.reg_dist = FixedDistribution(self.reg_mean)

    # Evaluation (non-trauma only)
    self.exam_dist = FixedDistribution(self.exam_mean)

    # Trauma/stablisation duration (trauma only)
    self.trauma_dist = FixedDistribution(self.trauma_mean)

    # Non-trauma treatment
    self.nt_treat_dist = FixedDistribution(self.non_trauma_treat_mean)

    # treatment of trauma patients
    self.treat_dist = FixedDistribution(self.trauma_treat_mean)

    # probability of non-trauma patient requiring treatment
    self.nt_p_treat_dist = FixedDistribution(self.non_trauma_treat_p)

    # probability of non-trauma versus trauma patient
    self.p_trauma_dist = FixedDistribution(self.prob_trauma)

    # init sampling for non-stationary poisson process
    # we leave this as stochastic.
    self.init_nspp()


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize(
    "random_no_set",
    [
        (42),
        (1),
        (754),
        (9876534321),
        (76546783986555),
    ],
)
def test_deterministic_nontrauma_pathway(random_no_set):
    """
    Test a single_run of the model when the non_trauma
    pathway is set to determinstic values with no capacity
    constraints.

    Arrivals are left as stochastic, this should not affect
    result.

    Expected result:
    ---------------
        total time in system = 37.3 allowing for floating point error.
    """
    # create a new scenario and set prob of trauma to 100%
    M = 1000_000_000

    # create a new scenario and set prob of trauma to 100%
    scenario = Scenario(
        n_triage=M,
        n_reg=M,
        n_exam=M,
        n_trauma=M,
        n_cubicles_1=M,
        n_cubicles_2=M,
        non_trauma_treat_p=1.0,
        prob_trauma=0.0,
    )

    # calculate the expected time in system = sum of deterministic activity times.
    expected_total_time = (
        scenario.triage_mean
        + scenario.reg_mean
        + scenario.exam_mean
        + scenario.non_trauma_treat_mean
    )

    # overwrite the function
    scenario.init_sampling = types.MethodType(init_sampling_to_fixed, scenario)

    # run the determinstic pathway (random numbers should make no difference)
    run_results = single_run(scenario, random_no_set=random_no_set)

    # test
    assert (
        pytest.approx(run_results["05_total_time(non-trauma)"].iloc[0])
        == expected_total_time
    )


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize(
    "random_no_set",
    [
        (42),
        (1),
        (754),
        (9876534321),
        (76546783986555),
    ],
)
def test_deterministic_trauma_pathway(random_no_set):
    """
    Test a single_run of the model when the trauma
    pathway is set to determinstic values with no capacity
    constraints.

    Arrivals are left as stochastic, this should not affect
    result.

    Expected result:
    ---------------
        total time in system = sum(triage+stablisation+treatment)
        allowing for floating point error.
    """
    # create a new scenario and set prob of trauma to 100%
    M = 1000_000_000

    # create a new scenario and set prob of trauma to 100%
    scenario = Scenario(
        n_triage=M,
        n_reg=M,
        n_exam=M,
        n_trauma=M,
        n_cubicles_1=M,
        n_cubicles_2=M,
        prob_trauma=1.0,
    )

    # calculate the expected time in system = sum of deterministic activity times.
    expected_total_time = (
        scenario.triage_mean + scenario.trauma_mean + scenario.trauma_treat_mean
    )

    # overwrite the function
    scenario.init_sampling = types.MethodType(init_sampling_to_fixed, scenario)

    # run the determinstic pathway (random numbers should make no difference)
    run_results = single_run(scenario, random_no_set=random_no_set)

    # test
    assert (
        pytest.approx(run_results["08_total_time(trauma)"].iloc[0])
        == expected_total_time
    )

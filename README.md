GENIE Worker Scoring
====================
*automatic crowdsourcing spammer detection*

This repository presents code related GENIE, a system for automatic
human evaluation of text generation tasks (see [the paper](#citation)
for further description of GENIE). In particular, it offers tools for
automatically detecting spammers in crowdsourcing based on their answers
to test question.


Overview
--------
To monitor the quality of crowd work, it's common to embed test
questions into tasks. The person requesting the work already knows the
test questions' answers, so they can automatically determine if a
question was answered wrong; however, knowing the number of correctly
answered questions still leaves open whether the worker happened to
answer a question wrong, or provided arbitrary answers. Given the number
of correctly answered questions, the `workerscoring` package in this
repo infers the probability that a worker is a spammer.

### How It Works

We formalize the scenario with a generative model of how workers answer
test questions. Each worker has some fixed probability, `p`, of
answering a test question correctly. These probabilities are distributed
according to a prior.

In the *threshold* model, we define any worker who's probability is at
or below a user-defined threshold, `p <= threshold`, as a spammer. In
the *component* model, we model the prior as a mixture of two or more
groups, where any worker not in the highest performing group is
considered to be a spammer.

At a technical level, we use an empirical Bayesian approach. The model
fits the prior via maximum likelihood, and then computes the posterior
of `p` for each worker. Finally, the model returns the posterior
probabilities that each worker meets the definition of being a spammer.

To handle multiple test question types, you can compute whether each was
answered correctly and then group all the statistics together.
Alternatively, you could compute the probability that a worker is a
spammer for ANY of the question types, and use that as an overall
spammer probability. In other words, if `q1` and `q2` are the
probability of spamming on question types one and two, then the overall
probability is `1 - (1 - q1) * (1 - q2)`.

This overall approach is inherently sequential and automatically
calibrates itself to the data. Thus, using these models you can
automatically identify spammers, at a configurable level of certainty,
as soon as the data provides sufficient evidence to support a decision.

### Core Interface

The main classes are the `workerscoring.detectors.ThresholdDetector` and
`workerscoring.detectors.ComponentDetector` which respectively implement
the threshold and component definitions of spammers. To use either,
begin with arrays containing the successes and failures for each worker:

    # successes : num_workers-length array of ints
    successes = np.array([1, 10, 2, ...])

    # failures : num_workers-length array of ints
    failures = np.array([9, 0, 1, ...])

Then, use the `fit` class method to instantiate and fit the model:

    from workerscoring import detectors

    detector = detectors.ThresholdDetector.fit(
      successes=successes,
      failures=failures,
      threshold=0.9,
    )

Once fitting finishes, you can then predict spammer probabilities on
either the same or even new data:

    spammer_probabilities = detector.predict(successes, failures)

You can then choose some cut off probability beyond which to declare
someone a spammer. Since the method is inherently sequential, it's good
to set a high cut off (e.g., 0.999) and then wait until you're
sufficiently confident before making a decision. In practice, you can
usually reach high confidence with only a few test questions.


Setup
-----
This software was developed with Python 3.7.

### Installation

Install using `pip`:

    pip install git+https://github.com/allenai/genie-worker-scoring.git

If installing from source locally, and you plan to edit the code,
consider using `pip`'s `--editable` flag.

### Tests

Run the tests with `unittest`. From the repository root, execute:

    python -m unittest


Citation
--------
GENIE was originally presented in the paper [*GENIE: A Leaderboard for
Human-in-the-Loop Evaluation of Text Generation*][genie-paper]. Please cite it
as follows:

    @misc{khashabi-etal-2021-genie,
      title = {
        GENIE: A Leaderboard for Human-in-the-Loop Evaluation of Text Generation
      },
      author = {
        Khashabi, Daniel
        and Stanovsky, Gabriel
        and Bragg, Jonathan
        and Lourie, Nicholas
        and Kasai, Jungo
        and Choi, Yejin
        and Smith, Noah A.
        and Weld, Daniel S.
      },
      publisher = {arXiv},
      year = {2021},
      url = {https://arxiv.org/abs/2101.06561},
      doi = {10.48550/ARXIV.2101.06561},
      keywords = {
        Computation and Language (cs.CL),
        Artificial Intelligence (cs.AI),
        FOS: Computer and information sciences,
        FOS: Computer and information sciences
      },
      copyright = {arXiv.org perpetual, non-exclusive license}
    }


[genie-paper]: https://arxiv.org/abs/2101.06561

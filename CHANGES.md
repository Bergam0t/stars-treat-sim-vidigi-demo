# CHANGELOG

## v1.1.0

### Updates and new features

* Upgraded internal implementation of generating non-overlapping random number streams. This is now implemented to use `np.random.SeedSequence`. See https://numpy.org/doc/stable/reference/random/parallel.html

### Patches

* `setup.py` now links to correct Github URL
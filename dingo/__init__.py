# imoprting structures:
# grism -> utils -> galaxy -> kinematics

# set up version
try:
    from setuptools_scm import get_version
    __version__ = get_version(root='..', relative_to=__file__)
except Exception:
    __version__ = "unknown"

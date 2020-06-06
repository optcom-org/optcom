import pytest

import optcom.utils.constants as cst
from optcom.components.abstract_pass_comp import AbstractPassComp
from optcom.components.abstract_start_comp import AbstractStartComp
from optcom.components.gaussian import Gaussian
from optcom.components.ideal_coupler import IdealCoupler
from optcom.domain import Domain
from optcom.field import Field
from optcom.layout import Layout, DelError, LinkError, PropagationError,\
    WrongPortWarning, StartSimError, SelfLinkError


# ----------------------------------------------------------------------
# Tests ----------------------------------------------------------------
# ----------------------------------------------------------------------

@pytest.mark.layout
def test_link_to_itself():
    """Should fail if component's port tries to link to itself."""
    # Environment creation
    a = IdealCoupler(name='a')
    layout = Layout()
    # Testing
    pytest.raises(SelfLinkError, layout.link, (a[1], a[1]))
    pytest.raises(SelfLinkError, layout.link, (a[1], a[0]))


@pytest.mark.layout
def test_already_linked():
    """Should fail if component's port is already linked."""
    # Environment creation
    a = IdealCoupler(name='a')
    b = IdealCoupler(name='b')
    c = IdealCoupler(name='c')
    layout = Layout()
    layout.link((a[1],b[0]))
    # Testing
    pytest.raises(LinkError, layout.link, (c[1], b[0]))
    pytest.raises(LinkError, layout.link, (b[0], c[2]))
    pytest.raises(LinkError, layout.link, (a[1], b[0]))


@pytest.mark.layout
def test_link_unidir_to_itself():
    """Should fail if component's port tries to link to itself."""
    # Environment creation
    a = IdealCoupler(name='a')
    layout = Layout()
    # Testing
    pytest.raises(LinkError, layout.link_unidir, (a[1], a[1]))
    pytest.raises(LinkError, layout.link_unidir, (a[1], a[0]))


@pytest.mark.layout
def test_already_unidir_linked():
    """Should fail if component's port is already linked."""
    # Environment creation
    a = IdealCoupler(name='a')
    b = IdealCoupler(name='b')
    c = IdealCoupler(name='c')
    layout = Layout()
    layout.link_unidir((a[1],b[0]))
    # Testing
    pytest.raises(LinkError, layout.link_unidir, (c[1], b[0]))
    pytest.raises(LinkError, layout.link_unidir, (b[0], c[2]))
    pytest.raises(LinkError, layout.link_unidir, (a[1], b[0]))


@pytest.mark.layout
def test_non_existant_link_del():
    """Should raise error if trying to delete a non-existing link."""
    # Environment creation
    a = IdealCoupler(name='a')
    b = IdealCoupler(name='b')
    c = IdealCoupler(name='c')
    layout = Layout()
    layout.link((a[1],b[0]), (a[2], b[1]))
    layout.link_unidir((b[2],c[2]), (a[0], c[1]))
    layout.del_link((c[2],b[2]))  # Valid (even if unidir in other dir.)
    # Testing
    pytest.raises(DelError, layout.del_link, (c[2], b[2]))
    pytest.raises(DelError, layout.del_link, (a[1], b[1]))
    pytest.raises(DelError, layout.del_link, (a[2], c[2]))


@pytest.mark.layout
def test_degree():
    """Should be of a specific degree for specific tree."""
    # Environment creation
    a = IdealCoupler(name='a')
    b = IdealCoupler(name='b')
    c = IdealCoupler(name='c')
    d = IdealCoupler(name='d')
    e = IdealCoupler(name='e')
    f = IdealCoupler(name='f')
    layout = Layout()
    layout.link((a[0],b[0]), (a[1],c[0]), (b[1], d[0]), (b[2], e[0]),
                (c[1], a[2]))
    # Testing
    assert layout.get_degree(a) == 3
    assert layout.get_degree(b) == 3
    assert layout.get_degree(c) == 2
    assert layout.get_degree(d) == 1
    assert layout.get_degree(e) == 1
    assert layout.get_degree(f) == 0


@pytest.mark.layout
def test_leafs():
    """Should be specific leafs for specific tree."""
    # Environment creation
    a = IdealCoupler(name='a')
    b = IdealCoupler(name='b')
    c = IdealCoupler(name='c')
    d = IdealCoupler(name='d')
    e = IdealCoupler(name='e')
    f = IdealCoupler(name='f')
    layout = Layout()
    layout.link((a[0],b[0]), (a[1],c[0]), (b[1], d[0]), (b[2], e[0]))
    # Testing
    assert layout.get_leafs_of_comps([a,b,c,d,e,f]) == [c,d,e]
    assert layout.leafs == [c,d,e]


@pytest.mark.layout
def test_wrong_start_comp_output():
    """Should fail if the right error is not raised."""
    # Environment creation
    class DummyStartComp(AbstractStartComp):
        def __init__(self):
            super().__init__('', '', [cst.OPTI_ALL], True)
        def __call__(self, domain):

            return ([0, 0], [Field(Domain(), 1)])

    start = DummyStartComp()
    a = IdealCoupler(name='a')
    layout = Layout()
    layout.link((start[0], a[0]))
    # Testing
    pytest.raises(PropagationError, layout.run, start)


@pytest.mark.layout
def test_wrong_start_comp_ports():
    """Should fail if the right warning is not shown."""
    # Environment creation
    class DummyStartComp(AbstractStartComp):
        def __init__(self):
            super().__init__('', '', [cst.OPTI_ALL], True)
        def __call__(self, domain):

            return ([4], [Field(Domain(), 1)])

    start = DummyStartComp()
    a = IdealCoupler(name='a')
    layout = Layout()
    layout.link((start[0], a[0]))
    # Testing
    pytest.warns(WrongPortWarning, layout.run, start)


@pytest.mark.layout
def test_wrong_pass_comp_output():
    """Should fail if the right error is not raised."""
    # Environment creations
    class DummyPassComp(AbstractPassComp):
        def __init__(self):
            super().__init__('', '', [cst.OPTI_ALL, cst.OPTI_ALL], True)
        def __call__(self, domain, ports, fields):

            return ([1 for i in range(len(fields)+1)], fields)

    start = Gaussian()
    pass_comp = DummyPassComp()
    a = IdealCoupler()
    layout = Layout()
    layout.link((start[0], pass_comp[0]), (pass_comp[1], a[0]))
    # Testing
    pytest.raises(PropagationError, layout.run, start)


@pytest.mark.layout
def test_wrong_pass_comp_ports():
    """Should fail if the right error is not raised."""
    # Environment creations
    class DummyPassComp(AbstractPassComp):
        def __init__(self):
            super().__init__('', '', [cst.OPTI_ALL, cst.OPTI_ALL], True)
        def __call__(self, domain, ports, fields):

            return ([5 for i in range(len(fields))], fields)

    start = Gaussian()
    pass_comp = DummyPassComp()
    a = IdealCoupler()
    layout = Layout()
    layout.link((start[0], pass_comp[0]), (pass_comp[1], a[0]))
    # Testing
    pytest.warns(WrongPortWarning, layout.run, start)


@pytest.mark.layout
def test_wrong_start_comp_for_run():
    """Should fail if the layout is start with a wrong component."""
    # Environment creations
    start = Gaussian()
    a = IdealCoupler()
    layout = Layout()
    layout.link((a[0], start[0]))
    # Testing
    pytest.raises(StartSimError, layout.run, a)

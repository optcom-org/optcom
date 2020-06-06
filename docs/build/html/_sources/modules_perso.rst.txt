Modules Description
===================

This document will drive you through the architecture of Optcom.

Simulation Framework
--------------------

The Layout, Domain, Field and Component are the building blocks of Optcom's optical system simulation framework.

Field
^^^^^
Fields represent an electric or optical signal. The object Field can contain multiple channels and save the values of the electro-magnetic field envelopes as well as other physic's characteristics. Moreover, a variety of helper functions are available for Field objects.

Component
^^^^^^^^^
Components represent electric or optical physical block such as laser, fiber and so on. There are two types of components in Optcom. First, ``StartComp`` which create a Field object and can launch the simulation. Second, ``PassComp`` which receive Field objects, transform it, and pass it along to the next component. A component is composed of ports.

Domain
^^^^^^
The domain contains information, i.e. physic's parameters, that will be shared by all components.

Layout
^^^^^^
A Layout allow to build a system by connecting the components to each other via their ports. Morevore, the Layout is managing the propagation of the Domain and Fields through the system.


Simulation Tools
----------------

Constraints
^^^^^^^^^^^
The Constraint objects represent constraints that the layout must comply whith while propagating Field objects in the Layout.

Effects
^^^^^^^
The Effect object represent electric / optical effect that can be used to define equations.

Equations
^^^^^^^^^
The Equation object is used to define equations that need a numerical solver and which describe the Field object transformation in a component.

Parameters
^^^^^^^^^^
The Parameter object is a standalone object which can be used as a helper object. It describes a physic's parameter such as the refractive index.

Solvers
^^^^^^^
The solver object is used to numerically solve the Equation objects.

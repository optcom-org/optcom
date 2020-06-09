# Copyright 2019 The Optcom Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""".. moduleauthor:: Sacha Medaer"""

import warnings
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

import optcom.config as cfg
import optcom.utils.constants as cst
import optcom.utils.utilities as util
from optcom.components.abstract_component import AbstractComponent
from optcom.components.abstract_pass_comp import AbstractPassComp
from optcom.components.abstract_start_comp import AbstractStartComp
from optcom.components.port import Port
from optcom.constraints.abstract_constraint import AbstractConstraint
from optcom.constraints.constraint_coprop import ConstraintCoprop
from optcom.constraints.constraint_max_pass_port import ConstraintMaxPassPort
from optcom.constraints.constraint_port_in import ConstraintPortIn
from optcom.constraints.constraint_port_valid import ConstraintPortValid
from optcom.constraints.constraint_waiting import ConstraintWaiting
from optcom.domain import Domain
from optcom.field import Field


# Exceptions
class LayoutError(Exception):
    pass

class LinkError(LayoutError):
    pass

class UnknownLinkError(LinkError):
    pass

class SelfLinkError(LinkError):
    pass

class DelError(LayoutError):
    pass

class StartSimError(LayoutError):
    pass

class PropagationError(LayoutError):
    pass

class LayoutWarning(UserWarning):
    pass

class WrongPortWarning(LayoutWarning):
    pass

class StartCompInputWarning(LayoutWarning):
    pass


class Layout(object):
    """Represent an optical system.

    The layout is represented as a graph, i.e. set of components
    linked together by either unidirectionnal or bidirectionnal edges.
    The components contain the information on their neighbors, i.e. to
    which components and ports they are linked to.

    Attributes
    ----------
    domain : optcom.domain.Domain
        The domain which the layout is bound to.
    name : str
        The name of the Layout.

    """

    def __init__(self, domain: Domain = Domain(), name: str = 'Layout'
                 ) -> None:
        """
        Parameters
        ----------
        domain : optcom.domain.Domain
            The domain which the layout is bound to.
        name :
            The name of the layout.

        """
        # Attr types check ---------------------------------------------
        util.check_attr_type(domain, 'Domain', Domain)
        util.check_attr_type(name, 'name', str)
        # Attr ---------------------------------------------------------
        self._nbr_comps: int = 0
        self._comps: List[AbstractComponent] = []
        self.domain: Domain = domain
        self.name: str = name
        # N.B.: List[int] is dummy list to inc/dec the unique int in it
        coprop = ConstraintCoprop()
        waiting = ConstraintWaiting()
        port_in = ConstraintPortIn()
        port_valid = ConstraintPortValid()
        max_pass_port = ConstraintMaxPassPort()
        self._constraints: List[AbstractConstraint] = [coprop, waiting,
                                                       port_in, port_valid,
                                                       max_pass_port]
    # ==================================================================
    # In-build methos ==================================================
    # ==================================================================
    def __str__(self):
        str_to_return: str = ""
        if (self._comps):
            str_to_return += "Structure of layout '{}':\n\n".format(self.name)
            for comp in self._comps:
                for port in comp:
                    if (not port.is_free()):
                        str_to_return += str(port) + '\n'
        else:
            str_to_return += "Layout '{}' is empty".format(self.name)

        return str_to_return
    # ==================================================================
    # Getters and setters ==============================================
    # ==================================================================
    @property
    def leafs(self) -> List[AbstractComponent]:

        return self.get_leafs_of_comps(self._comps)
    # ==================================================================
    # Print helpers ====================================================
    # ==================================================================
    def print_propagation(self, comp: AbstractComponent, comp_port_id: int,
                          ngbr: AbstractComponent, ngbr_port_id: int) -> None:
        util.print_terminal("Component '{}' has sent a signal from "
                            "port {} to port {} of component '{}'."
                            .format(comp.name, comp_port_id,
                                    ngbr_port_id, ngbr.name))
    # ==================================================================
    # Graph layout management ==========================================
    # ==================================================================
    def add_comp(self, *comps: AbstractComponent) -> None:
        """Add a component in the Layout.

        Parameters
        ----------
        comps : AbstractComponent
            A series of components to be added.

        """
        for comp in comps:
            if (comp not in self._comps):
                self._comps.append(comp)
                self._nbr_comps += 1
    # ==================================================================
    def _add_edge(self, port_1: Port, port_2: Port, unidir: bool = False
                  ) -> None:
        """Add a new edge in the Layout

        The edge can be either unidirectionnal or bidirectionnal.  In
        case of unidirectionnal edge, only the first component is linked
        to the second one.

        Parameters
        ----------
        port_1 : Port
            The first port of the edge.
        port_2 : Port
            The second port of the edge.
        unidir :
            If True, unidirectionnal link.

        """
        # Adding new edge ----------------------------------------------
        free_ports = port_1.is_free() and port_2.is_free()
        if (free_ports and port_1.comp != port_2.comp):
            self.add_comp(port_1.comp, port_2.comp)
            port_1.link_to(port_2)
            if (unidir):    # Undirected edge
                port_2.link_unidir_to(port_1)
            else:   # Directed edge
                port_2.link_to(port_1)
        # Edge can not be added ----------------------------------------
        else:
            error_message: str = ("Linking of component '{}' and component "
                "'{}' has encountered a problem.\n"
                .format(port_1.comp.name, port_2.comp.name))
            if (port_1.comp == port_2.comp):

                raise SelfLinkError(error_message +
                                    "Component '{}' can not be linked to "
                                    "itself.".format(port_1.comp.name))
            else:

                raise LinkError(error_message +
                                "The current states of those ports are:\n"
                                + str(port_1) + str(port_2))
    # ==================================================================
    def _del_edge(self, port_1: Port, port_2: Port) -> None:
        """Delete an edge in the Layout.

        Parameters
        ----------
        port_1 : Port
            The first port of the edge.
        port_2 : Port
            The second port of the edge.

        """
        link = (port_1.is_linked_to(port_2) and port_2.is_linked_to(port_1))
        # Link suppression ---------------------------------------------
        if (link):
            port_1.reset()
            port_2.reset()
        # Link does not exist ------------------------------------------
        else:
            error_msg: str = ("Can not delete a nonexistent edge from "
                "port {} of component '{}' to port {} of component '{}', "
                "action aborted.\n"
                .format(port_1.comp_port_id, port_1.comp.name,
                        port_2.comp_port_id, port_2.comp.name))
            error_msg += "The current states of those ports are:\n"

            raise DelError(error_msg + str(port_1) + str(port_2))
    # ==================================================================
    def add_link(self, port_1: Port, port_2: Port) -> None:
        """Add one bidirectionnal edge in the Layout.

        Parameters
        ----------
        port_1 :
            First port to be linked.
        port_2 :
            Second port to be linked.

        """

        self._add_edge(port_1, port_2, False)
    # ==================================================================
    def add_links(self, *links: Tuple[Port, Port]) -> None:
        """Add a series of bidirectionnal edges in the Layout.

        Parameters
        ----------
        links :
            Each tuple of the list contains the ports to be linked.

        """

        for edge in links:
            self._add_edge(edge[0], edge[1], False)
    # ==================================================================
    def add_unidir_link(self, port_1: Port, port_2: Port) -> None:
        """Add one unidirectionnal edge in the Layout

        Parameters
        ----------
        port_1 :
            First port to be linked.
        port_2 :
            Second port to be linked.

        """
        self._add_edge(port_1, port_2, True)
    # ==================================================================
    def add_unidir_links(self, *links: Tuple[Port, Port]) -> None:
        """Add a series of unidirectionnal edges in the Layout

        Parameters
        ----------
        links :
            Each tuple of the list contains the ports to be linked.

        """
        for edge in links:
            self._add_edge(edge[0], edge[1], True)
    # ==================================================================
    def del_link(self, port_1: Port, port_2: Port) -> None:
        """Delete a series of edges in the Layout.

        Parameters
        ----------
        port_1 :
            First port in the edge to be deleted.
        port_2 :
            Second port in the edge to be deleted.

        """
        self._del_edge(port_1, port_2)
    # ==================================================================
    def del_links(self, *links: Tuple[Port, Port]) -> None:
        """Delete a series of edges in the Layout.

        Parameters
        ----------
        links:
            Each tuple of the list contains the ports of the link to be
            deleted.

        """
        for edge in links:
            self._del_edge(edge[0], edge[1])
    # ==================================================================
    def reset(self) -> None:
        """Reset (i.e. empty) the Layout."""

        for comp in self._comps:
            for port in comp:
                port.reset()
        self._comps = []
        self._nbr_comps = 0
        # Reset structure data
        for constraint in self._constraints:
            constraint.reset()
    # ==================================================================
    # Constraints management ===========================================
    # ==================================================================
    def _update_constraints(self, comp: AbstractComponent, port_ids: List[int],
                            fields: List[Field]) -> None:
        """Update the constraint information."""
        for constraint in self._constraints:
            constraint.update(comp, port_ids, fields)
    # ==================================================================
    def _comply_with_constraints(self, comp: AbstractComponent,
                                 comp_port_id: int, ngbr: AbstractComponent,
                                 ngbr_port_id: int, field: Field
                                 ) -> Tuple[List[int], List[Field]]:
        """Return the port ids and fields to propagate in the component
        with respect to the constraints.
        """
        ngbr_port_ids: List[int] = []
        fields: List[Field] = []
        flag: bool = True
        # Check for respect of constraint ------------------------------
        for constraint in self._constraints:
            # Must first compute flag_constraint to run it, if make
            # flag = flag and call(), if flag is False, dont run call()
            flag_constraint = constraint.is_respected(comp, comp_port_id, ngbr,
                                                      ngbr_port_id, field)
            flag = flag and flag_constraint
        # All flags green -> can propagate -----------------------------
        if (flag):
            for constraint in self._constraints:
                ports_, fields_ = constraint.get_compliance(comp, comp_port_id,
                                                            ngbr, ngbr_port_id,
                                                            field)
                # Two diff constraints could have kept the same field
                # to restitue at the same time for diff reasons -> avoid
                # conflicts by checking if field already in output field
                for i in range(len(fields_)):
                    # Can not use 'not in' bcs of __eq__ Field overload
                    in_list: bool = False
                    j: int = 0
                    while (not in_list and j < len(fields)):
                        in_list = in_list or (fields[j] is fields_[i])
                        j += 1
                    if (not in_list):
                        fields.append(fields_[i])
                        ngbr_port_ids.append(ports_[i])
            fields.append(field)
            ngbr_port_ids.append(ngbr_port_id)

        return ngbr_port_ids, fields
    # ==================================================================
    # Prapagation management ===========================================
    # ==================================================================
    def must_save_output_field_of_comp(self, comp: AbstractComponent) -> bool:
        """Save the output fields of a component if requested by the
        user or if the component is a leaf.
        """

        return comp.save or (cfg.SAVE_LEAF_FIELDS and self.is_comp_leaf(comp))
    # ==================================================================
    def must_save_input_field_of_comp(self, comp: AbstractComponent) -> bool:
        """Save the input fields of a component if requested by the
        user.
        """

        return comp.save
    # ==================================================================
    def get_valid_nbrg(self, comp: AbstractComponent, comp_port_id: int
                       ) -> Optional[AbstractPassComp]:
        """Return the neighbor of the component at port id comp_port_id
        if the neighbor is of type AbstractPassComp.
        """
        potential_ngbr: Optional[AbstractComponent] = comp[comp_port_id].ngbr
        if (isinstance(potential_ngbr, AbstractPassComp)):

            return potential_ngbr
        else:
            if (isinstance(potential_ngbr, AbstractStartComp)):
                warning_message: str = ("The starter component {} can not "
                    "accept incoming fields.".format(potential_ngbr.name))
                warnings.warn(warning_message, StartCompInputWarning)

        return None
    # ==================================================================
    def _propagate_field(self, comp: AbstractComponent, comp_port_id: int,
                         field: Field) -> None:
        """Propagate one Field field from the port comp_port_id
        of component comp to the neighbor component of comp. Check if
        all the constraints are respected before propagation.

        Parameters
        ----------
        comp : optcom.components.abstract_component.AbstractComponent
            The component from which the field to propagate comes from.
        comp_port_id :
            The id port of the component from which the field to
            propagate comes from.
        field : optcom.field.Field
            The field to propagate.

        """
        comp_port: Port = comp[comp_port_id]
        # Recording output field ---------------------------------------
        if (self.must_save_output_field_of_comp(comp)):
            comp_port.save_field(field)
        # Propagate field to neighbor of comp --------------------------
        if (comp_port.is_valid_for_propagation()):
            ngbr: Optional[AbstractPassComp] = None
            ngbr = self.get_valid_nbrg(comp, comp_port_id)
            if (ngbr is not None):
                ngbr_port_id: int = comp_port.ngbr_port_id
                ngbr_port: Port = ngbr[ngbr_port_id]
                # Recording input field --------------------------------
                if (self.must_save_input_field_of_comp(ngbr)):
                    ngbr_port.save_field(field)
                # Check for constraints compliance ---------------------
                self.print_propagation(comp, comp_port_id, ngbr, ngbr_port_id)
                ngbr_port_ids, fields = self._comply_with_constraints(comp,
                                       comp_port_id, ngbr, ngbr_port_id, field)
                # Valid propagation management -------------------------
                if (fields):
                    output_port_ids, output_fields = ngbr(self.domain,
                                                          ngbr_port_ids,
                                                          fields)
                    # Recursive call through propagate()
                    self._propagate(ngbr, output_port_ids, output_fields)
    # ==================================================================
    def _propagate(self, comp: AbstractComponent, output_port_ids: List[int],
                   output_fields: List[Field]) -> None:
        # Check for valid outputs --------------------------------------
        if (len(output_port_ids) != len(output_fields)):

            raise PropagationError("The length of the output_ports list and "
                "output_fields list must be equal.")
        output_port_ids_: List[int] = []
        output_fields_: List[Field] = []
        for i in range(len(output_port_ids)):
            if (not comp.is_port_id_valid(output_port_ids[i])):
                if (output_port_ids[i] >=0):  # if < 0, convention for no prop.
                    warning_message: str = ("The port number {} for component "
                        "'{}' is not valid. Fields are not propagated."
                        .format(output_port_ids[i], comp.name))
                    warnings.warn(warning_message, WrongPortWarning)
            else:
                output_port_ids_.append(output_port_ids[i])
                output_fields_.append(output_fields[i])
        self._update_constraints(comp, output_port_ids_, output_fields_)
        # Propagate ----------------------------------------------------
        for i in range(len(output_port_ids_)):
            self._propagate_field(comp, output_port_ids_[i], output_fields_[i])
    # ==================================================================
    def run(self, *starters: AbstractStartComp) -> None:
        """Launch the simulation.

        Parameters
        ----------
        starter_comps : optcom.components.abstract_start_comp.AbstractStartComp
            The components from which the simulation starts.

        """
        # Check if valid component to start ----------------------------
        for starter in starters:
            if (isinstance(starter, AbstractStartComp)):
                self.add_comp(starter) # If not already in layout
            else:

                raise StartSimError("The component '{}' is not a valid "
                    "component to start the simulation with."
                    .format(starter.name))
        # Begin simulation ---------------------------------------------
        for starter in starters:
            output_port_ids, output_fields = starter(self.domain)
            self._propagate(starter, output_port_ids, output_fields)
    # ==================================================================
    def run_all(self) -> None:
        """Launch the simulation with all component of type
        AbstractStartComp.
        """
        starters: List[AbstractStartComp] = self.get_start_components()
        self.run(*starters)
    # ==================================================================
    # Graph structure properties =======================================
    # ==================================================================
    def is_comp_leaf(self, comp: AbstractComponent) -> bool:

        return comp in self.leafs
    # ==================================================================
    def get_start_components(self) -> List[AbstractStartComp]:
        """Retrun a list of components that are valid starter component,
        i.e. component that can be pass to the method run() to start the
        simulation.
        """
        starters: List[AbstractStartComp] = []
        for comp in self._comps:
            if (isinstance(comp, AbstractStartComp)):
                starters.append(comp)

        return starters
    # ==================================================================
    @staticmethod
    def get_leafs_of_comps(comps: List[AbstractComponent]
                           ) -> List[AbstractComponent]:
        """Return the leafs in the list of Component comps.

        Parameters
        ----------
        comps :
            Components to be checked if they are leafs.

        Returns
        -------
        :
            Components which are leaf.

        """
        leafs: List[AbstractComponent] = []
        for comp in comps:
            if (Layout.get_degree(comp) == 1):
                leafs.append(comp)

        return leafs
    # ==================================================================
    @staticmethod
    def get_degree(comp: AbstractComponent) -> int:
        """Return the degree of the component.

        Parameters
        ----------
        comp : AbstractComponent
            The component from which the degree must be calculated.

        Returns
        -------
        :
            Degree of the component given as parameter.

        """
        degree: int = 0
        if (len(comp) == 1):
            degree = 1
        else:
            for port in comp:
                if (not port.is_free() and port.is_type_out()):
                    degree += 1

        return degree

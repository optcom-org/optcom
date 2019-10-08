# This# This file is part of Optcom.
#
# Optcom is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Optcom is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Optcom.  If not, see <https://www.gnu.org/licenses/>.

""".. moduleauthor:: Sacha Medaer"""

from typing import Dict, List, Optional, Tuple

import numpy as np

import optcom.utils.constants as cst
import optcom.utils.utilities as util
from optcom.components.abstract_component import AbstractComponent
from optcom.components.abstract_pass_comp import AbstractPassComp
from optcom.components.abstract_start_comp import AbstractStartComp
from optcom.domain import Domain
from optcom.field import Field

Comp = AbstractComponent
Comp_Item = Tuple[AbstractComponent, int]
Link = Tuple[Comp_Item, Comp_Item]


class Layout(object):
    """Represent an optical system.

    The layout is represented as a graph, i.e. set of components
    linked together by either unidirectionnal or bidirectionnal edges.
    The components contain the information on their neighbors, i.e. to
    which components and ports they are linked to.

    Attributes
    ----------
    domain : Domain
        The domain which the layout is bound to.
    name : str
        The name of the Layout.

    """

    def __init__(self, domain: Domain = Domain(), name: str = 'Layout'
                 ) -> None:
        """
        Parameters
        ----------
        domain : Domain
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
        self._stack_coprop: Dict[Comp_Item, Tuple[List[Field], List[int]]] = {}
        self._stack_waiting: Dict[Comp, Tuple[List[int], List[Field]]] = {}
        self._stack_wait_policy: Dict[Comp, List[int]] = {}
        self._leaf_comps: List[AbstractComponent] = []
        self._constraint_to_check = ["coprop", "waiting", "port_in",
                                     "port_valid", "max_pass"]
    # ==================================================================
    def __str__(self):

        if (self._comps):
            util.print_terminal("Structure of layout '{}':".format(self.name))
            for comp in self._comps:
                for port_nbr in range(len(comp)):
                    if (comp.get_neighbor(port_nbr) is not None):
                            comp.print_port_state(port_nbr)
        else:
            util.print_terminal("Layout '{}' is empty".format(self.name))

        return str()
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
    def _add_edge(self, comp_1: AbstractComponent, port_comp_1: int,
                  comp_2: AbstractComponent, port_comp_2: int,
                  unidir: bool = False) -> None:
        """Add a new edge in the Layout

        The edge can be either unidirectionnal or bidirectionnal.  In
        case of bidirectionnal edge, only the first component is linked
        to the second one.

        Parameters
        ----------
        comp_1 : AbstractComponent
            The first component of the edge.
        port_comp_1 :
            The port of the first component.
        comp_2 : AbstractComponent
            The second component of the edge.
        port_comp_2 :
            The port of the second component.
        unidir :
            If True, unidirectionnal link.

        """
        # Adding new edge ----------------------------------------------
        if (comp_1.is_port_free(port_comp_1)
                and comp_2.is_port_free(port_comp_2) and comp_1 != comp_2):
            self.add_comp(comp_1)
            self.add_comp(comp_2)
            comp_1.link_to(port_comp_1, comp_2, port_comp_2)
            if (unidir):    # Undirected edge
                comp_2.link_to(port_comp_2, comp_1, cst.UNIDIR_PORT)
            else:   # Directed edge
                comp_2.link_to(port_comp_2, comp_1, port_comp_1)
        # Edge can not be added ----------------------------------------
        else:
            util.warning_terminal("Linking of component '{}' and component "
                "'{}' has encountered a problem, action aborted:"
                .format(comp_1.name, comp_2.name))
            if (comp_1 == comp_2):
                util.print_terminal("Component '{}' can not be linked to "
                    "itself".format(comp_1.name), '')
            else:
                comp_1.print_port_state(port_comp_1)
                comp_2.print_port_state(port_comp_2)
    # ==================================================================
    def _del_edge(self, comp_1: AbstractComponent, port_comp_1: int,
                  comp_2: AbstractComponent, port_comp_2: int) -> None:
        """Delete an edge in the Layout.

        Parameters
        ----------
        comp_1 : AbstractComponent
            The first component of the edge.
        port_comp_1 : int
            The port of the first component.
        comp_2 : AbstractComponent
            The second component of the edge.
        port_comp_2 : int
            The port of the second component.

        """
        link = (comp_1.is_linked_to(port_comp_1, comp_2, port_comp_2)
                and comp_2.is_linked_to(port_comp_2, comp_1, port_comp_1))
        link_unidir = (comp_1.is_linked_to(port_comp_1, comp_2, port_comp_2)
                       and comp_2.is_linked_unidir_to(port_comp_2, comp_1))
        # Link suppression ---------------------------------------------
        if (link or link_unidir):
            comp_1.del_port(port_comp_1)
            comp_2.del_port(port_comp_2)
        # Link does not exist ------------------------------------------
        else:
            util.warning_terminal("Can not delete a nonexistent edge from "
                "port {} of component '{}' to port {} of component '{}', "
                "action aborted:"
                .format(port_comp_1, comp_1.name, port_comp_2, comp_2.name))
            comp_1.print_port_state(port_comp_1)
            comp_2.print_port_state(port_comp_2)
    # ==================================================================
    def link(self, *links: Link) -> None:
        """Add a series of bidirectionnal edges in the Layout.

        Parameters
        ----------
        links :
            Each tuple of the list contains the components to be linked
            as well as their respective ports.

        """

        for edge in links:
            self._add_edge(edge[0][0], edge[0][1], edge[1][0],
                           edge[1][1], False)
    # ==================================================================
    def link_unidir(self, *links: Link) -> None:
        """Add a series of unidirectionnal edges in the Layout

        Parameters
        ----------
        links :
            Each tuple of the list contains the components to be linked
            as well as their respective ports.

        """
        for edge in links:
            self._add_edge(edge[0][0], edge[0][1], edge[1][0],
                           edge[1][1], True)
    # ==================================================================
    def del_link(self, *links: Link) -> None:
        """Delete a series of edges in the Layout.

        Parameters
        ----------
        links:
            Each tuple of the list contains the components to be linked
            as well as their respective ports.

        """
        for edge in links:
            self._del_edge(edge[0][0], edge[0][1], edge[1][0], edge[1][1])
    # ==================================================================
    def reset(self) -> None:
        """Reset (i.e. empty) the Layout."""

        for comp in self._comps:
            for port_nbr in range(len(comp)):
                comp.del_port(port_nbr)
        self._comps = []
        self._nbr_comps = 0
        # Reset structure data
        self._stack_coprop = {}
        self._stack_waiting = {}
        self._leaf_comps = []
    # ==================================================================
    # Constraints management ===========================================
    # ==================================================================
    def _update_constraints(self, comp: AbstractComponent, ports: List[int],
                            fields: List[Field]) -> None:
        for constraint in self._constraint_to_check:
            getattr(self, "_update_{}".format(constraint))(comp, ports, fields)
    # ==================================================================
    def _comply_with_constraints(self, comp: AbstractComponent,
                                 neighbor: AbstractComponent, port: int,
                                 field: Field, input_port: int
                                 ) -> Tuple[List[int], List[Field]]:
        input_ports: List[int] = []
        fields: List[Field] = []
        flag: bool = True
        for constraint in self._constraint_to_check:
            # !! must first compute flag_constraint to run it, if make
            # flag = flag and call(), if flag is False, dont run call()
            flag_constraint = getattr(self, "_respect_{}".format(constraint))(
                                      comp, neighbor, port, field, input_port)
            flag = flag and flag_constraint

        if (flag):    # All flags green -> can propagate
            for constraint in self._constraint_to_check:
                ports_, fields_ = getattr(self, "_get_{}".format(constraint))(
                                          comp, neighbor, port, field,
                                          input_port)
                fields.extend(fields_)
                input_ports.extend(ports_)
            fields.append(field)
            input_ports.append(input_port)

        return input_ports, fields
    # ==================================================================
    # Copropagating fields constraint ----------------------------------
    def _update_coprop(self, comp: AbstractComponent, ports: List[int],
                       fields: List[Field]) -> None:
        """Update copropagating fields."""
        for i, port in enumerate(ports):
            if (ports.count(port) > 1): # >1 ouptut field from same port
                if (self._stack_coprop.get((comp, port)) is not None):
                    self._stack_coprop[(comp, port)][0].append(fields[i])
                    self._stack_coprop[(comp, port)][1][0] += 1
                else:
                    self._stack_coprop[(comp, port)] = ([fields[i]], [1])
    # ==================================================================
    def _respect_coprop(self, comp: AbstractComponent,
                        neighbor: AbstractComponent, port: int, field: Field,
                        input_port: int) -> bool:
        """Check if need to wait for other copropagating fields."""
        flag = True
        if (self._stack_coprop.get((comp, port)) is not None):
            self._stack_coprop[(comp, port)][1][0] -= 1
            if (self._stack_coprop[(comp, port)][1][0] > 0):
                flag = False
                util.print_terminal("Signal is waiting for copropagating "
                    "fields.", '')

        return flag
    # ==================================================================
    def _get_coprop(self, comp: AbstractComponent, neighbor: AbstractComponent,
                    port: int, field: Field, input_port: int
                    ) -> Tuple[List[int], List[Field]]:
        ports: List[int] = []
        fields: List[Field] = []
        if (self._stack_coprop.get((comp, port)) is not None):
            # The last field should be the current one, debug: (unitest)
            if (field != self._stack_coprop[(comp, port)][0][-1]):
                util.warning_terminal("The last field of coprop stack should "
                    "be the current field.")
            fields = self._stack_coprop[(comp, port)][0][:-1]
            ports = [input_port for i in range(len(fields))]
            self._stack_coprop.pop((comp, port))

        return ports, fields
    # ==================================================================
    # Waiting constraint -----------------------------------------------
    def _update_waiting(self, comp: AbstractComponent, ports: List[int],
                        fields: List[Field]) -> None:
        return None
    # ==================================================================
    def _respect_waiting(self, comp: AbstractComponent,
                         neighbor: AbstractComponent, port: int, field: Field,
                         input_port: int) -> bool:
        flag = True
        wait_policy: List[List[int]] = neighbor.get_wait_policy(input_port)
        if (wait_policy):
            if (self._stack_waiting.get(neighbor) is None):
                self._stack_waiting[neighbor] = ([input_port], [field])
            else:
                self._stack_waiting[neighbor][0].append(input_port)
                self._stack_waiting[neighbor][1].append(field)
            # If more than one policy matches, take the first one
            flag = False
            i = 0
            while (i < len(wait_policy) and not flag):
                port_count = [self._stack_waiting[neighbor][0].count(wait_port)
                              for wait_port in wait_policy[i]]
                if (0 not in port_count):
                    flag = True
                    self._stack_wait_policy[neighbor] = wait_policy[i]
                i += 1
            if (not flag):
                util.print_terminal("Signal is waiting for fields "
                    "arriving at other ports.", '')

        return flag
    # ==================================================================
    def _get_waiting(self, comp: AbstractComponent,
                     neighbor: AbstractComponent, port: int, field: Field,
                     input_port: int) -> Tuple[List[int], List[Field]]:
        ports: List[int] = []
        fields: List[Field] = []
        if (self._stack_waiting.get(neighbor) is not None):
            # The last field should be the current one, debug: (unitest)
            if (field != self._stack_waiting[neighbor][1][-1]):
                util.warning_terminal("The last field of coprop stack should "
                    "be the current field.")
            wait_policy = self._stack_wait_policy[neighbor]
            #debug
            if (wait_policy is None):
                util.warning_terminal("wait_policy shouldn't be None if "
                    "self._stack_waiting.get(neighbor) is not None")
            ports_ = []
            fields_ = []
            for i, elem in enumerate(self._stack_waiting[neighbor][0][:-1]):
                if (elem in wait_policy):
                    ports.append(elem)
                    fields.append(self._stack_waiting[neighbor][1][i])
                else:
                    ports_.append(elem)
                    fields_.append(self._stack_waiting[neighbor][1][i])
            self._stack_waiting[neighbor] = (ports_, fields_)
            # Clear variables
            if (not self._stack_waiting[neighbor][0]):
                self._stack_waiting.pop(neighbor)
            self._stack_wait_policy.pop(neighbor)

        return ports, fields
    # ==================================================================
    # Port in constraint ===============================================
    def _update_port_in(self, comp: AbstractComponent, ports: List[int],
                        fields: List[Field]) -> None:
        return None
    # ==================================================================
    def _respect_port_in(self, comp: AbstractComponent,
                         neighbor: AbstractComponent, port: int, field: Field,
                         input_port: int) -> bool:
        """Check if the port allows an input."""
        flag = neighbor.is_port_type_in(input_port)
        if (not flag):
            util.warning_terminal( "Port {} of component {} does not "
                "accept input, field will be ignored."
                .format(input_port, neighbor.name), '')

        return flag
    # ==================================================================
    def _get_port_in(self, comp: AbstractComponent,
                     neighbor: AbstractComponent, port: int, field: Field,
                     input_port: int) -> Tuple[List[int], List[Field]]:

        return [], []
    # ==================================================================
    # Port valid constraint ============================================
    def _update_port_valid(self, comp: AbstractComponent, ports: List[int],
                           fields: List[Field]) -> None:
        return None
    # ==================================================================
    def _respect_port_valid(self, comp: AbstractComponent,
                            neighbor: AbstractComponent, port: int,
                            field: Field, input_port: int) -> bool:
        """Check if type of field correspond to type of port."""
        flag = neighbor.is_port_type_valid(input_port, field)
        if (not flag):
            util.warning_terminal("Wrong field type to enter port {} "
                "of component {}, field will be ignored."
                .format(input_port, neighbor.name), '')

        return flag
    # ==================================================================
    def _get_port_valid(self, comp: AbstractComponent,
                        neighbor: AbstractComponent, port: int, field: Field,
                        input_port: int) -> Tuple[List[int], List[Field]]:

        return [], []
    # ==================================================================
    # Max pass constraint ==============================================
    def _update_max_pass(self, comp: AbstractComponent, ports: List[int],
                         fields: List[Field]) -> None:
        return None
    # ==================================================================
    def _respect_max_pass(self, comp: AbstractComponent,
                          neighbor: AbstractComponent, port: int, field: Field,
                          input_port: int) -> bool:
        """Check if the number of time a field can pass by input_port of
        component neighbor is not overpassed.
        """
        neighbor.inc_counter_pass(input_port)
        flag = neighbor.is_counter_below_max_pass(input_port)
        if (not flag):
            util.warning_terminal("Max number of times a field can go through "
                "port {} of component '{}' has been reached, field will be "
                "ignored."
                .format(input_port, neighbor.name), '')

        return flag
    # ==================================================================
    def _get_max_pass(self, comp: AbstractComponent,
                      neighbor: AbstractComponent, port: int, field: Field,
                      input_port: int) -> Tuple[List[int], List[Field]]:

        return [], []
    # ==================================================================
    # Field time window management =====================================
    # ==================================================================
    def _check_field_time_overlap(self, fields):
        """Check if the time window of the fields overlap."""
        return True
    def _match_field_time(self, fields):
        """Extend temporarily the time window of all fields to make them
        completely overlapping.
        """
        return None
    def _reset_field_time(self, fields, field_time_data):
        """Reset the time window as they were before extending them."""
        return None
    # ==================================================================
    # Prapagation management ===========================================
    # ==================================================================
    def _init_propagation(self, comp: AbstractComponent, output_port: int,
                          output_field: Field) -> None:
        """Propagate one Field"""
        # Recording field ----------------------------------------------
        if (comp.save or (comp in self._leaf_comps)):
            comp.save_field(output_port, output_field)
        # Propagate output_field to neighbors of comp ------------------
        neighbor: Optional[AbstractPassComp] = None
        potential_neighbor = comp.get_neighbor(output_port)
        if (isinstance(potential_neighbor, AbstractPassComp)): # no starter
            neighbor = potential_neighbor
        if (neighbor is not None):
            input_port_neighbor = comp.get_port_neighbor(output_port)
            if (input_port_neighbor != cst.UNIDIR_PORT):
                if (neighbor.save):       # Recording field
                    neighbor.save_field(input_port_neighbor, output_field)
                # Valid propagation management -----------------------------
                util.print_terminal("Component '{}' has sent a signal from "
                                    "port {} to port {} of component '{}'."
                                    .format(comp.name, output_port,
                                            input_port_neighbor,
                                            neighbor.name))
                input_ports_neighbor, output_fields = \
                    self._comply_with_constraints(comp, neighbor, output_port,
                                                  output_field,
                                                  input_port_neighbor)
                if (output_fields):            # Constraints respected
                    if (self._check_field_time_overlap(output_fields)):
                        # Make sure time windows overlap if >1 field
                        field_time_data = self._match_field_time(output_fields)
                        # Send the fields into the component
                        output_ports_neighbor, output_fields_neighbor =\
                            neighbor(self.domain, input_ports_neighbor,
                                     output_fields)
                        # Reset original time window
                        self._reset_field_time(output_fields, field_time_data)
                        # Recursive function
                        self._propagate(neighbor, output_ports_neighbor,
                                        output_fields_neighbor)
                    else:     # Time window of output_fields not overlapping
                        util.warning_terminal("Fields can not be accepted at "
                            "the entrance of component '{}' as their time "
                            "windows are not overlapping."
                            .format(neighbor.name), '')
    # ==================================================================
    def _propagate(self, comp: AbstractComponent, output_ports: List[int],
                   output_fields: List[Field]) -> None:
        self._update_constraints(comp, output_ports, output_fields)
        # Debug
        if (len(output_ports) != len(output_fields)):
            util.warning_terminal("The length of the output_ports list and "
                "output_fields list must be equal.")
        # Propagate
        for i in range(len(output_ports)):
            if (output_ports[i] != -1):     # Explicit no propag. port policy
                self._init_propagation(comp, output_ports[i], output_fields[i])
    # ==================================================================
    def run(self, *starter_comps: AbstractStartComp) -> None:
        """Launch the simulation.

        Parameters
        ----------
        starter_comps : AbstractStartComp
            The components from which the simulation starts.

        """
        for starter in starter_comps:
            self.add_comp(starter)
        self._get_structure()
        for starter in starter_comps:
            # TO DO: check if starter is valid start pulse
            output_ports, output_fields = starter(self.domain)
            self._propagate(starter, output_ports, output_fields)
    # ==================================================================
    # Graph structure properties =======================================
    # ==================================================================
    def _get_structure(self) -> None:
        """Draw structural constraints from the Layout."""
        self._leaf_comps = self.get_leafs(self._comps)
    # ==================================================================
    @staticmethod
    def get_leafs(comps: List[AbstractComponent]) -> List[AbstractComponent]:
        """Return the leafs of the Layout in the list given as parameter

        Parameters
        ----------
        comps :
            Components to be checked if they are leafs of the Layout.

        Returns
        -------
        :
            Components which are leafs of the Layout.

        """
        leafs = []
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
            The component to be checked

        Returns
        -------
        :
            Degree of the component given as parameter.

        """
        degree = 0
        if (len(comp) == 1):
            degree = 1
        else:
            for port in range(len(comp)):
                if (not comp.is_port_free(port)
                        and comp.is_port_type_out(port)):
                    degree += 1

        return degree


if __name__ == "__main__":

    from optcom.components.abstract_component import AbstractComponent

    # Create a layout with six components
    default_name = 'dflt'
    ports_type = [cst.OPTI_ALL, cst.OPTI_ALL, cst.OPTI_ALL, cst.OPTI_ALL]
    save = False
    save_all = False
    a = AbstractComponent('a', default_name, ports_type, save, save_all)
    b = AbstractComponent('b', default_name, ports_type, save, save_all)
    c = AbstractComponent('c', default_name, ports_type, save, save_all)
    d = AbstractComponent('d', default_name, ports_type, save, save_all)
    e = AbstractComponent('e', default_name, ports_type, save, save_all)
    f = AbstractComponent('f', default_name, ports_type, save, save_all)

    lt = Layout()

    print("########################## Check creation and suppression of edges")
    lt.link((a[0],b[0]))    # Valid
    lt.link((a[2],a[2]), (a[2],b[0]), (f[1], b[0]), (a[0],f[1]))  # Not valid
    lt.link((a[2],b[1]))    # Valid
    lt.link_unidir((a[3],a[1]),(a[3],b[0]))  # Not Valid
    lt.link_unidir((b[2],a[3]), (c[1],d[2]))  # Valid
    print(lt)
    lt.del_link((a[1],b[2]),(a[2],b[0]))  # Not valid
    lt.del_link((a[2],b[1]))    # Valid
    lt.del_link((a[3],b[2]))    # Not Valid
    lt.del_link((b[2],a[3]), (b[2],a[3]))    # Valid
    print(lt)
    lt.reset()
    print(lt)

    print(Layout.__doc__)

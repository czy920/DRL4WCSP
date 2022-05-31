import xml.etree.ElementTree as et
from xml.dom import minidom
import os, random

MODE_BINARY = 'BINARY'
MODE_NARY = 'NARY'
MODE_ASYM = 'ASYM'
MODE_ADV = 'ADV'


class Problem:

    def __init__(self):
        self.agents = dict()  # {agent_id: description}
        self.domains = dict()  # {domain_size: domain_id}
        self.agent_domain_mapping = dict()  # {agent_id: domain_size}
        self.constraints = dict()  # {scope: idx}
        self.functions = dict()  # {idx: obj}
        self.domain_idx = 1
        self.constraint_idx = 1

    def reset(self):
        self.agents = dict()  # {agent_id: description}
        self.domains = dict()  # {domain_size: domain_id}
        self.agent_domain_mapping = dict()  # {agent_id: domain_size}
        self.constraints = dict()  # {scope: idx}
        self.functions = dict()  # {idx: obj}
        self.domain_idx = 1
        self.constraint_idx = 1

    def random_meeting_scheduling(self, nb_people, nb_meeting, nb_slots, nb_select_meetings, min_travel_time, max_travel_time):
        self.reset()
        meetings = []
        for i in range(nb_meeting):
            self.add_agent(i + 1, nb_slots)
            meetings.append(set())
        cur_meeting = 1
        minus = True
        for i in range(nb_people):
            selected_meetings = set()
            for _ in range(nb_select_meetings):
                if cur_meeting <= nb_meeting:
                    rnd_meeting = cur_meeting
                    cur_meeting += 1
                else:
                    minus = False
                    rnd_meeting = random.randint(1, nb_meeting)
                while rnd_meeting in selected_meetings:
                    rnd_meeting = random.randint(1, nb_meeting)
                selected_meetings.add(rnd_meeting)
                meetings[rnd_meeting - 1].add(i)
            if minus:
                cur_meeting -= 1
        for i in range(nb_meeting - 1):
            for j in range(i + 1, nb_meeting):
                people = meetings[j].intersection(meetings[i])
                if len(people) != 0:
                    people = meetings[i].union(meetings[j])
                    travel_time = random.randint(min_travel_time, max_travel_time)
                    matrix = []
                    for val_i in range(nb_slots):
                        matrix.append([])
                        for val_j in range(nb_slots):
                            matrix[val_i].append(len(people) if abs(val_i - val_j) < travel_time else 0)
                    # print(max([max(matrix[i]) for i in range(len(matrix))]))
                    self.add_constraint([i + 1, j + 1], matrix)

    def random_binary(self, nb_agent, domain_size, p1, min_cost=0, max_cost=100, gc=False,
                      weighted=False, adv_domains=None, decimal=-1):
        if adv_domains is None:
            adv_domains = list()
        assert isinstance(nb_agent, int) and nb_agent > 0
        assert 0 < p1 <= 1
        assert 0 <= min_cost < max_cost
        assert isinstance(adv_domains, list)
        edge_cnt = int(nb_agent * (nb_agent - 1) / 2 * p1)
        assert len(adv_domains) <= edge_cnt
        self.reset()
        for i in range(nb_agent):
            self.add_agent(i + 1, domain_size)
        adv_agents = []
        for i, adv_size in enumerate(adv_domains):
            adv_id = len(self.agents) + 1
            adv_agents.append(adv_id)
            self.add_agent(adv_id, adv_size, 'ADV')
        connected = set()
        remaining = set([x for x in self.agents if self.agents[x] != 'ADV'])
        remaining_adv_agents = [i for i in range(len(adv_agents))]
        while len(remaining) > 0:
            agnt1 = random.sample(remaining, 1)[0]
            if len(connected) > 0:
                agnt2 = random.sample(connected, 1)[0]
                if len(adv_agents) == 0:
                    self.add_constraint([agnt1, agnt2], Problem._random_matrix(domain_size, domain_size,
                                                                               min_cost, max_cost, gc, weighted, decimal))
                else:
                    if len(remaining_adv_agents) > 0:
                        rand_adv_idx = random.randrange(0, len(remaining_adv_agents))
                        rand_adv_idx = remaining_adv_agents.pop(rand_adv_idx)
                    else:
                        rand_adv_idx = random.sample(range(len(adv_agents)), 1)[0]
                    domains = {agnt1: domain_size, agnt2: domain_size, adv_agents[rand_adv_idx]: adv_domains[rand_adv_idx]}
                    self.add_constraint([agnt1, agnt2, adv_agents[rand_adv_idx]], Problem._random_tensor(domains, min_cost, max_cost))
            remaining.discard(agnt1)
            connected.add(agnt1)
        edge_cnt -= (nb_agent - 1)
        remaining = set([x for x in self.agents if self.agents[x] != 'ADV'])
        while edge_cnt > 0:
            rand_adv_idx = -1
            if len(adv_agents) == 0:
                matrix = Problem._random_matrix(domain_size, domain_size, min_cost, max_cost, gc, weighted, decimal)
            else:
                if len(remaining_adv_agents) > 0:
                    rand_adv_idx = random.randrange(0, len(remaining_adv_agents))
                    rand_adv_idx = remaining_adv_agents.pop(rand_adv_idx)
                else:
                    rand_adv_idx = random.sample(range(len(adv_agents)), 1)[0]
                domains = {0: domain_size, 1: domain_size, adv_agents[rand_adv_idx]: adv_domains[rand_adv_idx]}
                matrix = Problem._random_tensor(domains, min_cost, max_cost)
            while True:
                agents = random.sample(remaining, 2)
                if len(adv_agents) > 0:
                    agents.append(adv_agents[rand_adv_idx])
                adv = None if rand_adv_idx == -1 else adv_agents[rand_adv_idx]
                if self.add_constraint(agents, matrix, adv):
                    break
            edge_cnt -= 1

    # def random_scale_free(self, nb_agent, domain_size, m1, m2, min_cost=0, max_cost=100, gc=False,
    #                       weighted=False, decimal=-1):
    #     assert isinstance(nb_agent, int) and nb_agent > 0
    #     assert isinstance(m1, int) and isinstance(m2, int)
    #     assert m2 <= m1 <= nb_agent
    #     self.reset()
    #     for i in range(nb_agent):
    #         self.add_agent(i + 1, domain_size)
    #     remaining = set(self.agents.keys())
    #     connected = dict()
    #     for _ in range(m1):
    #         agnt1 = random.sample(remaining, 1)[0]
    #         if len(connected) > 0:
    #             agnt2 = random.sample(set(connected.keys()), 1)[0]
    #             self.add_constraint([agnt1, agnt2], Problem._random_matrix(domain_size, domain_size,
    #                                                                        min_cost, max_cost, gc, weighted, decimal))
    #             connected[agnt2] += 1
    #         remaining.discard(agnt1)
    #         connected[agnt1] = 0 if len(connected) == 0 else 1
    #     while len(remaining) > 0:
    #         agnt1 = random.sample(remaining, 1)[0]
    #         c_agents = [[x, y] for x, y in connected.items()]
    #         cnt = [x[-1] for x in c_agents]
    #         for _ in range(m2):
    #             p_sum = sum(cnt)
    #             prob = [x / p_sum for x in cnt]
    #             idx = proportional_selection(prob)
    #             agnt2 = c_agents[idx][0]
    #             cnt[idx] = 0
    #             ok = self.add_constraint([agnt1, agnt2], Problem._random_matrix(domain_size, domain_size,
    #                                                                             min_cost, max_cost, gc, weighted, decimal))
    #             assert ok
    #             connected[agnt2] += 1
    #         remaining.discard(agnt1)
    #         connected[agnt1] = m2

    @classmethod
    def _random_matrix(cls, domain1, domain2, min_cost, max_cost, gc=False, weighted=False, decimal=-1):
        data = []
        for i in range(domain1):
            data.append([0] * domain2)
            for j in range(domain2):
                if not gc:
                    if decimal <= 0:
                        cost = random.randint(min_cost, max_cost)
                    else:
                        cost = random.random() * (max_cost - min_cost) + min_cost
                        cost = round(cost, decimal)
                elif not weighted:
                    cost = 0 if i != j else 1
                else:
                    cost = 0 if i != j else random.randint(min_cost, max_cost)
                data[i][j] = cost
        return data

    # @classmethod
    # def _random_tensor(cls, domains, min_cost, max_cost):
    #     # tensor = Tensor(domains)
    #     for i in range(len(tensor)):
    #         tensor.data[i] = random.randint(min_cost, max_cost)
    #     return tensor

    def add_agent(self, agent_id, domain_size, description=None):
        assert isinstance(agent_id, int) and agent_id not in self.agents
        assert isinstance(domain_size, int) and domain_size > 0
        if domain_size not in self.domains:
            self.domains[domain_size] = 'D{}'.format(self.domain_idx)
            self.domain_idx += 1
        self.agents[agent_id] = description if description else 'Agent {}'.format(agent_id)
        self.agent_domain_mapping[agent_id] = domain_size

    def add_constraint(self, scope, function, adv=None):
        if adv is not None:
            assert adv in scope
            r_scope = list(scope)
            r_scope.remove(adv)
            r_scope = tuple(sorted(r_scope))
            for s in self.constraints:
                if r_scope == s[:-1]:
                    return False
            scope = tuple(sorted(scope))
        else:
            scope = tuple(sorted(scope))
            if scope in self.constraints:
                return False
        self.constraints[scope] = 'C{}'.format(self.constraint_idx)
        self.functions['R{}'.format(self.constraint_idx)] = function
        self.constraint_idx += 1
        return True

    def save(self, path, mode=MODE_BINARY, meta_data=None):
        assert mode in [MODE_BINARY, MODE_ASYM, MODE_NARY, MODE_ADV]
        if not meta_data:
            meta_data = dict()
        else:
            assert isinstance(meta_data, dict)
        dir_name = os.path.dirname(path)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        if os.path.exists(path):
            os.remove(path)
        meta_data['type'] = mode
        root = et.Element('instance')
        et.SubElement(root, 'presentation', meta_data)
        agents = et.SubElement(root, 'agents', {'nbAgents': str(len(self.agents))})
        for agent_id in self.agents:
            et.SubElement(agents, 'agent', {'name': 'A{}'.format(agent_id), 'id': str(agent_id),
                                            'description': self.agents[agent_id]})
        domains = et.SubElement(root, 'domains', {'nbDomains': str(len(self.domains))})
        for domain_size in self.domains:
            et.SubElement(domains, 'domain', {'name': self.domains[domain_size], 'nbValues': str(domain_size)})
        variables = et.SubElement(root, 'variables', {'nbVariables': str(len(self.agents))})
        for agent_id in self.agents:
            et.SubElement(variables, 'variable', {'agent': 'A{}'.format(agent_id), 'name': 'X{}.1'.format(agent_id),
                                                  'domain': self.domains[self.agent_domain_mapping[agent_id]],
                                                  'description': 'Variable X{}.1'.format(agent_id)})

        constraints = et.SubElement(root, 'constraints', {'nbConstraints': str(len(self.constraints))})
        for scp in self.constraints:
            if mode == MODE_ADV:
                checksum = [0 if self.agents[x] != 'ADV' else 1 for x in scp]
                assert checksum[-1] == 1 and sum(checksum) == 1
            scope_variables = ['X{}.1'.format(x) for x in scp]
            scope = ' '.join(scope_variables)
            arity = str(len(scp))
            et.SubElement(constraints, 'constraint', {'name': self.constraints[scp], 'arity': arity, 'scope': scope,
                                                      'reference': self.constraints[scp].replace('C', 'R')})

        relations = et.SubElement(root, 'relations', {'nbRelations': str(len(self.constraints))})
        for name in self.functions:
            e = et.SubElement(relations, 'relation', {'name': name})
            if mode != MODE_NARY and mode != MODE_ADV:
                matrix = self.functions[name]
                parts = []
                for row in range(len(matrix)):
                    for col in range(len(matrix[0])):
                        txt = '{}:{} {}'.format(matrix[row][col], row + 1, col + 1)
                        parts.append(txt)
                e.text = '|'.join(parts)
            else:
                tensor = self.functions[name]
                # assert isinstance(tensor, Tensor)
                relation_pth = os.path.join(os.path.dirname(path), 'relations', os.path.basename(path)[: -4])
                if not os.path.exists(relation_pth):
                    os.makedirs(relation_pth)
                relation_pth = os.path.join(relation_pth, name)
                f = open(relation_pth, 'w')
                data = []
                for d in tensor.data:
                    data.append(str(d))
                    if len(data) == 1000:
                        f.write('|'.join(data) + '\n')
                        data = []
                if len(data) != 0:
                    f.write('|'.join(data) + '\n')
                f.close()
        xmlstr = minidom.parseString(et.tostring(root)).toprettyxml(indent="   ")
        with open(path, "w") as f:
            f.write(xmlstr)
"""
Assumptions:
- We are testing with only one depth of inheritance
- The interface/superclass has the same frequency as the subclass or the implementer
- If we have a static method then it should have a class, we will check for it
"""

import json
import string
from typing import Any, Dict
import sys

# Reconfigure stdout to use utf-8 encoding
sys.stdout.reconfigure(encoding='utf-8')

class Sequencer:
    """Punctuations"""
    IMPLEMENTATION = "\u039B"  # Λ
    INHERITANCE = "\u03A9"     # Ω
    CONSTRUCTION = "\u03A3"    # Σ
    SETTER_GETTER = "\u03A6"   # Φ
    GENERAL_PROCESSING = "\u0393"  # Γ
    STATIC_CALL = "\u03A4"     # Τ
    CLONING_CALL = "\u039E"    # Ξ

    """Special objects"""
    SUPER_CLASS = "\u0394"     # Δ
    INTERFACE = "\u03A8"       # Ψ
    MAIN_OBJECT = "\u03A0"     # Π
    STATIC_OBJECT = "\u03F4"   # ϴ

    def __init__(self, jsonfile: str):
        self.jsonPath = jsonfile
        self.method_dictionary: Dict[str, str] = {}
        self.edge_dictionary: Dict[str, Dict[str, Any]] = {}
        self.objects_aphabets_dictionary: Dict[str, str] = {}
        self.text_sequence = ""

    def initiate(self):
        with open(self.jsonPath) as f:
            data = json.load(f)

        for line in data['objects']:
            if 'cluster' not in line['name']:
                self.method_dictionary[line['_gvid']] = line['name']

        for line in data['edges']:
            self.edge_dictionary[line['_gvid']] = {
                'tail': line['tail'],
                'head': line['head'],
                'ncalls': line['label']
            }

        tmp_object_list = []
        for item in self.method_dictionary.values():
            if '.' in item:
                obj = str(item).split('.')
                if obj[0] != '__main__':
                    tmp_object_list.append(obj[0])
            else:
                if item != '__main__':
                    tmp_object_list.append(item)

        unique_objects_names = set(tmp_object_list)
        alpha_list = list(string.ascii_uppercase)[:len(unique_objects_names)]

        self.objects_aphabets_dictionary = dict(zip(unique_objects_names, alpha_list))
        self.objects_aphabets_dictionary.update({
            "__main__": self.MAIN_OBJECT,
            "static_class": self.STATIC_OBJECT,
            "interface_class": self.INTERFACE,
            "super_class": self.SUPER_CLASS
        })

    def check_architecture(self, module: Any, cls: str) -> str:
        cls_names = []
        try:
            class_ = getattr(module, cls)
            for super_cls in class_.mro():
                if super_cls.__name__ != class_.__name__ and super_cls.__name__ != 'object':
                    cls_names.append(super_cls.__name__)
        except AttributeError:
            return ""

        if len(cls_names) == 0:
            return ""
        elif len(cls_names) == 1:
            return self.SUPER_CLASS
        else:
            return self.INTERFACE

    def method_to_pontuations(self, method: str, called_object: str) -> str:
        if called_object == 'static_class':
            return self.STATIC_CALL
        elif method == '__init__':
            return self.CONSTRUCTION
        elif method in ['clone', 'copy']:
            return self.CLONING_CALL
        elif "_id" in method:
            return self.SETTER_GETTER
        else:
            return self.GENERAL_PROCESSING

    def generate_sequence(self, module: Any):
        # The first thing the main object call itself using INIT_AM and its frequency
        token = [self.objects_aphabets_dictionary['__main__'], self.CONSTRUCTION, self.objects_aphabets_dictionary['__main__']]
        self.text_sequence = ''.join(token)

        for call in self.edge_dictionary.values():
            caller = self.method_dictionary[call['tail']]
            called = self.method_dictionary[call['head']]
            ncalls = call['ncalls']

            if caller == '__main__':
                called_parts = str(called).split('.')
                if len(called_parts) > 1:
                    called_object, called_by_method = called_parts
                else:
                    called_object, called_by_method = 'static_class', called_parts[0]

                for _ in range(int(ncalls)):
                    if called_by_method == '__init__':
                        checked = self.check_architecture(module, called_object)
                        if checked == self.INTERFACE:
                            token = [self.objects_aphabets_dictionary.get(called_object, 'X'), self.IMPLEMENTATION, self.objects_aphabets_dictionary["interface_class"]]
                            self.text_sequence += ''.join(token)
                        elif checked == self.SUPER_CLASS:
                            token = [self.objects_aphabets_dictionary.get(called_object, 'X'), self.INHERITANCE, self.objects_aphabets_dictionary["super_class"]]
                            self.text_sequence += ''.join(token)

                    token = [self.objects_aphabets_dictionary["__main__"], self.method_to_pontuations(called_by_method, called_object), self.objects_aphabets_dictionary.get(called_object, 'X')]
                    self.text_sequence += ''.join(token)
            else:
                called_parts = str(called).split('.')
                if len(called_parts) > 1:
                    called_object, called_by_method = called_parts
                else:
                    continue  # Skip this iteration if we can't unpack properly

                caller_object = str(caller).split('.')[0]

                for _ in range(int(ncalls)):
                    if called_by_method == '__init__':
                        checked = self.check_architecture(module, called_object)
                        if checked == self.INTERFACE:
                            token = [self.objects_aphabets_dictionary.get(called_object, 'X'), self.IMPLEMENTATION, self.objects_aphabets_dictionary["interface_class"]]
                            self.text_sequence += ''.join(token)
                        elif checked == self.SUPER_CLASS:
                            token = [self.objects_aphabets_dictionary.get(called_object, 'X'), self.INHERITANCE, self.objects_aphabets_dictionary["super_class"]]
                            self.text_sequence += ''.join(token)

                    token = [self.objects_aphabets_dictionary.get(caller_object, 'X'), self.method_to_pontuations(called_by_method, called_object), self.objects_aphabets_dictionary.get(called_object, 'X')]
                    self.text_sequence += ''.join(token)

        return self.text_sequence
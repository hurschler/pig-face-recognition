class PathSwitcher:
    """Switcher class (function) for changing paths of each network"""
    def __init__(self):
        self.cases = []
        self.case_matched = False

    def add_case(self, value, callback, breaks=True):
        self.cases.append({
            'value': value,
            'callback': callback,
            'breaks': breaks
        })

    def case(self, value):
        results = []
        for case in self.cases:
            if self.case_matched == True or value == case ['value']:
                self.case_matched = True
                results.append(case['callback']())
                if case['breaks']:
                    break
        self.case_matched = False
        return results

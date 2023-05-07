import unittest
from main import Neldermead, Squeeze, Points, CreatePoints, Function
import json
import numpy as np


class MyTestCase(unittest.TestCase):
    def test1(self):
        with open('settings.json') as j:
            file = json.load(j)
        n = file['n']
        f = file['f']
        alpha = file['alpha']
        betta = file['betta']
        gama = file['gama']
        eps = file['eps']
        steps = file['steps']
        if file['start_simplex']:
            simplex = file['simplex']
            start_simplex = Points(n, simplex)
        else:
            start_simplex = CreatePoints(n)
        f = Function(f, n)
        expected = np.around(np.array(file['answer']), 2)
        result = Neldermead(n, f, alpha, betta, gama, eps, steps, start_simplex)

        self.assertEqual(result.all(), expected.all())

    def test2(self):
        with open('settings1.json') as j:
            file = json.load(j)
        n = file['n']
        f = file['f']
        alpha = file['alpha']
        betta = file['betta']
        gama = file['gama']
        eps = file['eps']
        steps = file['steps']
        if file['start_simplex']:
            simplex = file['simplex']
            start_simplex = Points(n, simplex)
        else:
            start_simplex = CreatePoints(n)
        f = Function(f, n)
        expected = []
        answer = file['answer']
        for i in range(len(answer[0])):
            tmp = []
            tmp.append(answer[0][i])
            tmp.append(answer[1][i])
            expected.append(np.around(np.array(tmp), 2))

        result = Neldermead(n, f, alpha, betta, gama, eps, steps, start_simplex)
        ans = False
        for x in expected:
            if np.equal(x.all(), result.all()):
                ans = True
                break
        self.assertTrue(ans, True)


if __name__ == '__main__':
    unittest.main()

import pickle
p = pickle.load(open('xgboost_model.pkl', 'rb'))
print('Model artifact type:', type(p))
try:
    print('Keys:', list(p.keys()))
except Exception:
    print('Not a dict, repr:')
    print(repr(p))

import json

samples_json = open("PolynomRenderTest/samples.json", "r")
samples = json.load(samples_json)
print(samples)
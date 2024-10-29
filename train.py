from ultralytics import YOLO, checks, hub

checks()

hub.login("fba500f7b229869c21a3551518bde7fa9a0fc270f9")

model = YOLO("https://hub.ultralytics.com/models/eHhEpyCykqFHL8gf8vIo")
results = model.train()

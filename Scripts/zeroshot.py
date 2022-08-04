from transformers import pipeline

text = "he four little girls killed in the 1963 church bombing in Alabama, they were all 14 and younger. " \
       "The kids who marched in Birmingham's Children's Crusade were beaten and had dogs sicced on them. " \
       "They were as young as 12. And Emmett Till was 14. He has a memorial in the lynching room, " \
       "but he wouldn't even be allowed to go down there without a chaperone."

classifier = pipeline("zero-shot-classification")
classifier(
    text,
    candidate_labels=["happiness", "anger", "sadness"],
)
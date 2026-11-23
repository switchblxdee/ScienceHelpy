"""Stream names and their business events."""

EVENT_STREAMS = {
    "paperImportRequested": "paper-library",
    "paperAddedToLibrary": "paper-library",
    "paperImportFailed": "paper-library",
    "paperContentPrepared": "paper-preparation",
    "paperPreparationFailed": "paper-preparation",
    "paperMadeSearchable": "paper-preparation",
    "sourceSearchRequested": "source-discovery",
    "localSourcesFound": "source-discovery",
    "localSourcesNotFound": "source-discovery",
    "externalSourceSearchCompleted": "source-discovery",
    "sourceSearchFailed": "source-discovery",
    "conversationStarted": "research-assistance",
    "questionSubmitted": "research-assistance",
    "answerProvided": "research-assistance",
    "answerCouldNotBeProvided": "research-assistance",
    "conversationEnded": "research-assistance",
}

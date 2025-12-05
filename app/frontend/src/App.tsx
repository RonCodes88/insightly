import { useMemo, useState } from 'react'
import './App.css'

type Prediction = {
  rating: number
  confidence: number
}

type ModelOption = 'naive_bayes' | 'rnn'

const MODEL_OPTIONS: Array<{ label: string; value: ModelOption }> = [
  { label: 'Naive Bayes', value: 'naive_bayes' },
  { label: 'RNN', value: 'rnn' },
]

const clamp = (value: number, min: number, max: number) =>
  Math.min(max, Math.max(min, value))

const normalizeRating = (value: unknown) => {
  if (value === null || value === undefined) {
    return null
  }

  const numericValue = Number(value)
  if (Number.isNaN(numericValue)) {
    return null
  }

  return Math.round(clamp(numericValue, 1, 5))
}

const normalizeConfidence = (value: unknown) => {
  if (value === null || value === undefined) {
    return null
  }

  const numericValue = Number(value)
  if (Number.isNaN(numericValue)) {
    return null
  }

  const percent = numericValue <= 1 ? numericValue * 100 : numericValue
  return Math.round(clamp(percent, 0, 100))
}

function App() {
  const [reviewText, setReviewText] = useState('')
  const [selectedModel, setSelectedModel] = useState<ModelOption>('naive_bayes')
  const [prediction, setPrediction] = useState<Prediction | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const confidenceDescriptor = useMemo(() => {
    if (!prediction) {
      return 'Awaiting prediction'
    }

    if (prediction.confidence >= 90) {
      return 'The model is very confident in this rating.'
    }

    if (prediction.confidence >= 70) {
      return 'Solid confidence, but consider a quick review.'
    }

    if (prediction.confidence >= 50) {
      return 'Moderate confidence. Pair with a manual check.'
    }

    return 'Low confidence. Treat as a suggestion only.'
  }, [prediction])

  const handlePredict = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault()

    if (!reviewText.trim()) {
      setError('Please enter or paste a review before requesting a prediction.')
      setPrediction(null)
      return
    }

    setIsLoading(true)
    setError(null)

    try {
      const response = await fetch('http://localhost:8000/analyze_review', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          review_text: reviewText.trim(),
          model: selectedModel,
        }),
      })

      if (!response.ok) {
        const message = await response.text()
        throw new Error(message || 'Prediction request failed.')
      }

      const data = await response.json()

      const rating =
        normalizeRating(data.rating) ??
        normalizeRating(data.predicted_rating) ??
        normalizeRating(data.predicted_score) ??
        normalizeRating(data.score) ??
        normalizeRating(data.prediction)

      if (rating === null) {
        throw new Error('The service did not return a rating. Please try again.')
      }

      const confidence =
        normalizeConfidence(data.confidence) ??
        normalizeConfidence(data.probability) ??
        normalizeConfidence(data.certainty) ??
        0

      setPrediction({ rating, confidence })
    } catch (requestError) {
      setError(
        requestError instanceof Error
          ? requestError.message
          : 'Something went wrong. Please try again.',
      )
      setPrediction(null)
    } finally {
      setIsLoading(false)
    }
  }

  const handleSample = (text: string) => {
    setReviewText(text)
    setPrediction(null)
    setError(null)
  }

  return (
    <div className="app-shell">
      <main className="app-card">
        <header className="card-header">
          <p className="eyebrow">Insightly · Review intelligence</p>
          <h1>Predict the rating behind any customer review</h1>
          <p className="subtitle">
            Paste the review text, hit predict, and get a suggested 1–5 star rating with a
            confidence signal for your operations team.
          </p>
        </header>

        <form className="review-form" onSubmit={handlePredict}>
          <label htmlFor="reviewInput">Customer review</label>
          <textarea
            id="reviewInput"
            name="review"
            placeholder="Start typing or pick a sample review below..."
            value={reviewText}
            onChange={(event) => setReviewText(event.target.value)}
            disabled={isLoading}
            rows={7}
          />

          <div className="model-selector">
            <label htmlFor="modelSelect">Select model</label>
            <select
              id="modelSelect"
              value={selectedModel}
              onChange={(event) => setSelectedModel(event.target.value as ModelOption)}
              disabled={isLoading}
            >
              {MODEL_OPTIONS.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </div>

          <div className="form-footer">
            <div className="sample-chips" aria-label="Sample reviews">
              <button
                type="button"
                onClick={() =>
                  handleSample(
                    'Absolutely loved the stay! Staff went above and beyond and the room was spotless.',
                  )
                }
                disabled={isLoading}
              >
                Delight
              </button>
              <button
                type="button"
                onClick={() =>
                  handleSample(
                    'Service was slow and no one seemed interested in helping even after multiple requests.',
                  )
                }
                disabled={isLoading}
              >
                Frustration
              </button>
              <button
                type="button"
                onClick={() =>
                  handleSample(
                    'Product quality is okay, but the shipping delay and missing accessories were disappointing.',
                  )
                }
                disabled={isLoading}
              >
                Mixed
              </button>
            </div>

            <button className="primary" type="submit" disabled={isLoading}>
              {isLoading ? 'Analyzing…' : 'Predict rating'}
            </button>
          </div>
        </form>

        <section className="prediction-panel" aria-live="polite">
          {error && <p className="prediction-message error">{error}</p>}

          {!error && prediction && (
            <div className="prediction-result">
              <div className="rating-display">
                <div>
                  <span className="rating-value">{prediction.rating}</span>
                  <span className="rating-scale">/5</span>
                </div>
                <p className="rating-description">{confidenceDescriptor}</p>
              </div>

              <div className="confidence">
                <div className="confidence-labels">
                  <span>Confidence</span>
                  <span>{prediction.confidence}%</span>
                </div>
                <div className="confidence-bar" role="img" aria-label={`Confidence ${prediction.confidence}%`}>
                  <div
                    className="confidence-fill"
                    style={{ width: `${prediction.confidence}%` }}
                  />
                </div>
              </div>
            </div>
          )}

          {!error && !prediction && (
            <p className="prediction-message">
              Predictions will appear here with the suggested rating and a confidence gauge.
            </p>
          )}
        </section>
      </main>
    </div>
  )
}

export default App

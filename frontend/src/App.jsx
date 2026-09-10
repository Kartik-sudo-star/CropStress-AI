import { Routes, Route, Navigate } from 'react-router-dom'
import { Layout } from './components/Layout'
import { Home } from './pages/Home'
import { Analyze } from './pages/Analyze'
import { Processing } from './pages/Processing'
import { Result } from './pages/Result'
import { Explainability } from './pages/Explainability'
import { Risk } from './pages/Risk'
import { Report } from './pages/Report'
import { History } from './pages/History'
import { Analytics } from './pages/Analytics'
import { NotFound } from './pages/NotFound'

export function App() {
  return (
    <Routes>
      <Route path="/" element={<Layout />}>
        <Route index element={<Home />} />
        <Route path="analyze" element={<Analyze />} />
        <Route path="processing" element={<Processing />} />
        <Route path="result/:analysisId" element={<Result />} />
        <Route path="explainability/:analysisId" element={<Explainability />} />
        <Route path="risk/:analysisId" element={<Risk />} />
        <Route path="report/:analysisId" element={<Report />} />
        <Route path="history" element={<History />} />
        <Route path="analytics" element={<Analytics />} />
        <Route path="*" element={<NotFound />} />
      </Route>
    </Routes>
  )
}
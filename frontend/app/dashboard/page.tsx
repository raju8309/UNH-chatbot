"use client";

import React, { useState, useEffect } from "react";
import { createPortal } from "react-dom";

// Utility functions and components
const formatRunIdDate = (runId: string) => {
  if (runId && runId.length === 15 && runId.includes('_')) {
    const [datePart, timePart] = runId.split('_');
    const year = datePart.substring(0, 4);
    const month = datePart.substring(4, 6);
    const day = datePart.substring(6, 8);
    const hour = timePart.substring(0, 2);
    const minute = timePart.substring(2, 4);
    const parsedDate = new Date(`${year}-${month}-${day}T${hour}:${minute}:00`);
    return `${parsedDate.toLocaleDateString()} at ${parsedDate.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}`;
  }
  return 'Invalid date format';
};

const getTestRunDisplayNumber = (testRuns: any[], targetRunId: string) => {
  const index = testRuns.findIndex((tr: any) => tr.run_id === targetRunId);
  return index !== -1 ? testRuns.length - index : 1;
};

const haveSameGoldSet = (report1: any, report2: any) => {
  if (!report1?.predictions_data?.predictions || !report2?.predictions_data?.predictions) {
    return false;
  }
  
  const ids1 = new Set(report1.predictions_data.predictions.map((p: any) => p.id));
  const ids2 = new Set(report2.predictions_data.predictions.map((p: any) => p.id));
  
  // Check if both sets have the same size and contain the same IDs
  if (ids1.size !== ids2.size) {
    return false;
  }
  
  for (const id of ids1) {
    if (!ids2.has(id)) {
      return false;
    }
  }
  
  return true;
};

const getCategoryBadgeClasses = (category: string) => {
  // Generate consistent colors based on category string hash
  const hashCode = (str: string) => {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      const char = str.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash; // Convert to 32-bit integer
    }
    return Math.abs(hash);
  };
  
  const colors = [
    'bg-blue-100 text-blue-800',
    'bg-green-100 text-green-800', 
    'bg-purple-100 text-purple-800',
    'bg-orange-100 text-orange-800',
    'bg-red-100 text-red-800',
    'bg-indigo-100 text-indigo-800',
    'bg-yellow-100 text-yellow-800',
    'bg-pink-100 text-pink-800',
    'bg-gray-100 text-gray-800',
    'bg-teal-100 text-teal-800'
  ];
  
  return colors[hashCode(category) % colors.length];
};

const base_doc_id = (url: string) => {
  if (!url) return '';
  try {
    const urlObj = new URL(url);
    const pathParts = urlObj.pathname.split('/').filter(part => part);
    return pathParts[pathParts.length - 1] || 'unknown';
  } catch {
    return 'unknown';
  }
};

interface RetrievedDocumentsProps {
  retrievedIds: any[];
  title?: string;
}

const RetrievedDocuments: React.FC<RetrievedDocumentsProps> = ({ retrievedIds, title = "Retrieved Documents:" }) => {
  const [hoveredItem, setHoveredItem] = useState<{ item: any; position: { x: number; y: number } } | null>(null);

  const handleMouseEnter = (item: any, event: React.MouseEvent) => {
    const rect = event.currentTarget.getBoundingClientRect();
    setHoveredItem({
      item,
      position: {
        x: rect.left,
        y: rect.top - 10 // Position above the element
      }
    });
  };

  const handleMouseLeave = () => {
    setHoveredItem(null);
  };

  return (
    <div className="mt-4">
      <h4 className="font-semibold text-gray-800 mb-2">{title}</h4>
      <div className="flex flex-wrap gap-2">
        {retrievedIds.map((item: any, idx: number) => (
          <span
            key={idx}
            className="bg-gray-100 text-gray-700 px-2 py-1 rounded text-sm cursor-help"
            onMouseEnter={(e) => handleMouseEnter(item, e)}
            onMouseLeave={handleMouseLeave}
          >
            {base_doc_id(item.url)}#{item.idx}
          </span>
        ))}
      </div>

      {/* Portal for tooltip */}
      {hoveredItem && createPortal(
        <div 
          className="fixed bg-white text-black border-2 border-[var(--unh-blue)] text-xs rounded py-3 px-4 z-[9999] min-w-96 max-w-lg shadow-lg pointer-events-none"
          style={{
            left: hoveredItem.position.x,
            top: hoveredItem.position.y,
            transform: 'translateY(-100%)'
          }}
        >
          <div className="space-y-2">
            <div><strong>Rank:</strong> {hoveredItem.item.rank}</div>
            <div><strong>Score:</strong> {hoveredItem.item.score?.toFixed(3)}</div>
            <div><strong>Title:</strong> <span className="break-words">{hoveredItem.item.title}</span></div>
            <div><strong>URL:</strong> <span className="break-all text-xs">{hoveredItem.item.url}</span></div>
            <div><strong>Tier:</strong> {hoveredItem.item.tier_name}</div>
            {hoveredItem.item.text && (
              <div><strong>Text:</strong> <span className="break-words">{hoveredItem.item.text.length > 300 ? `${hoveredItem.item.text.substring(0, 300)}...` : hoveredItem.item.text}</span></div>
            )}
          </div>
        </div>,
        document.body
      )}
    </div>
  );
};

interface MetricCardProps {
  title: string;
  value: number;
  subtitle: string;
  isPercentage?: boolean;
}

const MetricCard: React.FC<MetricCardProps> = ({ title, value, subtitle, isPercentage = true }) => {
  // Determine color based on performance thresholds
  const getPerformanceColor = (value: number) => {
    const percentage = isPercentage ? value * 100 : value;
    if (percentage >= 70) return 'text-green-600';
    if (percentage >= 50) return 'text-yellow-600';
    return 'text-red-600';
  };

  const dynamicColor = getPerformanceColor(value);

  return (
    <div className="bg-gray-50 rounded-lg shadow-sm p-4">
      <h3 className="text-sm font-semibold text-gray-700 mb-2">{title}</h3>
      <p className={`text-2xl font-bold ${dynamicColor}`}>
        {isPercentage ? (value * 100).toFixed(1) + '%' : value.toFixed(1)}
      </p>
      <p className="text-xs text-gray-500 mt-1">{subtitle}</p>
    </div>
  );
};

interface CircularProgressProps {
  value: number;
  color: string;
  label: string;
}

const CircularProgress: React.FC<CircularProgressProps> = ({ value, color, label }) => {
  // Determine color based on performance thresholds
  const getPerformanceColor = (value: number) => {
    const percentage = value * 100;
    if (percentage >= 70) return '#059669'; // green-600
    if (percentage >= 50) return '#d97706'; // yellow-600
    return '#dc2626'; // red-600
  };

  const dynamicColor = getPerformanceColor(value);

  return (
    <div className="text-center">
      <div className="relative inline-flex items-center justify-center w-16 h-16">
        <svg className="w-16 h-16 transform -rotate-90" viewBox="0 0 36 36">
          <path
            d="m18,2.0845 a 15.9155,15.9155 0 0,1 0,31.831 a 15.9155,15.9155 0 0,1 0,-31.831"
            fill="none"
            stroke="#e5e7eb"
            strokeWidth="2"
          />
          <path
            d="m18,2.0845 a 15.9155,15.9155 0 0,1 0,31.831 a 15.9155,15.9155 0 0,1 0,-31.831"
            fill="none"
            stroke={dynamicColor}
            strokeWidth="2"
            strokeDasharray={`${value * 100}, 100`}
          />
        </svg>
        <span className={`absolute text-sm font-bold`} style={{ color: dynamicColor }}>
          {(value * 100).toFixed(0)}%
        </span>
      </div>
      <div className="text-xs text-gray-600 mt-1">{label}</div>
    </div>
  );
};

interface MetricsSummaryProps {
  summary: any;
}

const MetricsSummary: React.FC<MetricsSummaryProps> = ({ summary }) => (
  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4 mb-6">
    <MetricCard 
      title="BERTscore F1" 
      value={summary.bertscore_f1} 
      subtitle="Semantic similarity" 
    />
    <MetricCard 
      title="SBERT Cosine" 
      value={summary.sbert_cosine} 
      subtitle="Sentence similarity" 
    />
    <MetricCard 
      title="SBERT Chunk" 
      value={summary.sbert_cosine_chunk} 
      subtitle="Chunk similarity" 
    />
    <MetricCard 
      title="Recall@1" 
      value={summary["recall@1"]} 
      subtitle="Top result accuracy" 
    />
    <MetricCard 
      title="NDCG@3" 
      value={summary["ndcg@3"]} 
      subtitle="Ranking quality" 
    />
  </div>
);

interface NuggetMetricsProps {
  metrics: any;
}

const NuggetMetrics: React.FC<NuggetMetricsProps> = ({ metrics }) => (
  <div className="bg-gray-50 rounded-lg shadow-sm p-4 overflow-hidden">
    <h5 className="text-lg font-bold text-gray-800 mb-4">Nugget Metrics</h5>
    <div className="flex justify-around items-center">
      <CircularProgress value={metrics.nugget_precision} color="" label="Precision" />
      <CircularProgress value={metrics.nugget_recall} color="" label="Recall" />
      <CircularProgress value={metrics.nugget_f1} color="" label="F1 Score" />
    </div>
  </div>
);

interface RankingMetricsProps {
  metrics: any;
}

const RankingMetrics: React.FC<RankingMetricsProps> = ({ metrics }) => (
  <div className="bg-gray-50 rounded-lg shadow-sm p-4 overflow-hidden">
    <h5 className="text-lg font-bold text-gray-800 mb-4">Ranking Performance</h5>
    <div className="flex justify-around items-center">
      <CircularProgress value={metrics["recall@3"]} color="" label="Recall@3" />
      <CircularProgress value={metrics["recall@5"]} color="" label="Recall@5" />
      <CircularProgress value={metrics["ndcg@5"]} color="" label="NDCG@5" />
    </div>
  </div>
);

interface ComparisonMetricCardProps {
  title: string;
  value1: number;
  value2: number;
  subtitle: string;
  isPercentage?: boolean;
  runNumber1?: number;
  runNumber2?: number;
}

const ComparisonMetricCard: React.FC<ComparisonMetricCardProps> = ({ 
  title, value1, value2, subtitle, isPercentage = true, runNumber1, runNumber2 
}) => {
  // Convert to display values first if percentage
  const displayValue1 = isPercentage ? value1 * 100 : value1;
  const displayValue2 = isPercentage ? value2 * 100 : value2;
  
  // Calculate absolute difference (Run 2 - Run 1)
  const absoluteDiff = displayValue2 - displayValue1;
  const formatValue = (val: number) => isPercentage ? (val * 100).toFixed(1) + '%' : val.toFixed(1);
  
  // Determine color based on performance thresholds
  const getPerformanceColor = (value: number) => {
    const percentage = isPercentage ? value * 100 : value;
    if (percentage >= 70) return 'text-green-600';
    if (percentage >= 50) return 'text-yellow-600';
    return 'text-red-600';
  };

  const color1 = getPerformanceColor(value1);
  const color2 = getPerformanceColor(value2);
  
  return (
    <div className="bg-gray-50 rounded-lg shadow-sm p-4">
      <h3 className="text-sm font-semibold text-gray-700 mb-2">{title}</h3>
      <div className="flex items-center justify-between mb-2">
        <div className="text-center flex-1">
          <p className={`text-xl font-bold ${color1}`}>{formatValue(value1)}</p>
          <p className="text-xs text-gray-500">Test Run #{runNumber1 || 1}</p>
        </div>
        <div className="px-2">
          <span className="text-gray-400">vs</span>
        </div>
        <div className="text-center flex-1">
          <p className={`text-xl font-bold ${color2}`}>{formatValue(value2)}</p>
          <p className="text-xs text-gray-500">Test Run #{runNumber2 || 2}</p>
        </div>
      </div>
      <div className="text-center border-t pt-2">
        <p className={`text-sm font-medium ${
          absoluteDiff > 0 ? 'text-green-600' : absoluteDiff < 0 ? 'text-red-600' : 'text-gray-600'
        }`}>
          {absoluteDiff > 0 ? '+' : ''}{absoluteDiff.toFixed(1)}%
        </p>
        <p className="text-xs text-gray-500 mt-1">{subtitle}</p>
      </div>
    </div>
  );
};

interface ComparisonSummaryProps {
  summary1: any;
  summary2: any;
  runNumber1?: number;
  runNumber2?: number;
}

const ComparisonSummary: React.FC<ComparisonSummaryProps> = ({ summary1, summary2, runNumber1, runNumber2 }) => (
  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4 mb-6">
    <ComparisonMetricCard 
      title="BERTscore F1" 
      value1={summary1.bertscore_f1} 
      value2={summary2.bertscore_f1}
      subtitle="Semantic similarity" 
      runNumber1={runNumber1}
      runNumber2={runNumber2}
    />
    <ComparisonMetricCard 
      title="SBERT Cosine" 
      value1={summary1.sbert_cosine} 
      value2={summary2.sbert_cosine}
      subtitle="Sentence similarity" 
      runNumber1={runNumber1}
      runNumber2={runNumber2}
    />
    <ComparisonMetricCard 
      title="SBERT Chunk" 
      value1={summary1.sbert_cosine_chunk} 
      value2={summary2.sbert_cosine_chunk}
      subtitle="Chunk similarity" 
      runNumber1={runNumber1}
      runNumber2={runNumber2}
    />
    <ComparisonMetricCard 
      title="Recall@1" 
      value1={summary1["recall@1"]} 
      value2={summary2["recall@1"]}
      subtitle="Top result accuracy" 
      runNumber1={runNumber1}
      runNumber2={runNumber2}
    />
    <ComparisonMetricCard 
      title="NDCG@3" 
      value1={summary1["ndcg@3"]} 
      value2={summary2["ndcg@3"]}
      subtitle="Ranking quality" 
      runNumber1={runNumber1}
      runNumber2={runNumber2}
    />
  </div>
);

interface ComparisonNuggetMetricsProps {
  metrics1: any;
  metrics2: any;
}

const ComparisonNuggetMetrics: React.FC<ComparisonNuggetMetricsProps> = ({ metrics1, metrics2 }) => {
  const calculatePercentDiff = (val1: number, val2: number) => {
    // Convert to display values (percentages) and calculate absolute difference (Run 2 - Run 1)
    const displayVal1 = val1 * 100;
    const displayVal2 = val2 * 100;
    return displayVal2 - displayVal1;
  };

  // Determine color based on performance thresholds
  const getPerformanceColor = (value: number) => {
    const percentage = value * 100;
    if (percentage >= 70) return 'text-green-600';
    if (percentage >= 50) return 'text-yellow-600';
    return 'text-red-600';
  };

  return (
    <div className="bg-gray-50 rounded-lg shadow-sm p-4 overflow-hidden">
      <h5 className="text-lg font-bold text-gray-800 mb-4">Nugget Metrics</h5>
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <span className="text-sm font-medium text-gray-700 w-20">Precision</span>
          <div className="flex items-center flex-1 justify-center gap-4">
            <span className={`text-lg font-bold ${getPerformanceColor(metrics1.nugget_precision)}`}>{(metrics1.nugget_precision * 100).toFixed(1)}%</span>
            <span className="text-xs text-gray-500">vs</span>
            <span className={`text-lg font-bold ${getPerformanceColor(metrics2.nugget_precision)}`}>{(metrics2.nugget_precision * 100).toFixed(1)}%</span>
          </div>
          <span className={`text-sm font-medium w-20 text-right ${
            calculatePercentDiff(metrics1.nugget_precision, metrics2.nugget_precision) > 0 
              ? 'text-green-600' : calculatePercentDiff(metrics1.nugget_precision, metrics2.nugget_precision) < 0 
              ? 'text-red-600' : 'text-gray-600'
          }`}>
            {calculatePercentDiff(metrics1.nugget_precision, metrics2.nugget_precision) > 0 ? '+' : ''}
            {calculatePercentDiff(metrics1.nugget_precision, metrics2.nugget_precision).toFixed(1)}%
          </span>
        </div>
        <div className="flex items-center justify-between">
          <span className="text-sm font-medium text-gray-700 w-20">Recall</span>
          <div className="flex items-center flex-1 justify-center gap-4">
            <span className={`text-lg font-bold ${getPerformanceColor(metrics1.nugget_recall)}`}>{(metrics1.nugget_recall * 100).toFixed(1)}%</span>
            <span className="text-xs text-gray-500">vs</span>
            <span className={`text-lg font-bold ${getPerformanceColor(metrics2.nugget_recall)}`}>{(metrics2.nugget_recall * 100).toFixed(1)}%</span>
          </div>
          <span className={`text-sm font-medium w-20 text-right ${
            calculatePercentDiff(metrics1.nugget_recall, metrics2.nugget_recall) > 0 
              ? 'text-green-600' : calculatePercentDiff(metrics1.nugget_recall, metrics2.nugget_recall) < 0 
              ? 'text-red-600' : 'text-gray-600'
          }`}>
            {calculatePercentDiff(metrics1.nugget_recall, metrics2.nugget_recall) > 0 ? '+' : ''}
            {calculatePercentDiff(metrics1.nugget_recall, metrics2.nugget_recall).toFixed(1)}%
          </span>
        </div>
        <div className="flex items-center justify-between">
          <span className="text-sm font-medium text-gray-700 w-20">F1 Score</span>
          <div className="flex items-center flex-1 justify-center gap-4">
            <span className={`text-lg font-bold ${getPerformanceColor(metrics1.nugget_f1)}`}>{(metrics1.nugget_f1 * 100).toFixed(1)}%</span>
            <span className="text-xs text-gray-500">vs</span>
            <span className={`text-lg font-bold ${getPerformanceColor(metrics2.nugget_f1)}`}>{(metrics2.nugget_f1 * 100).toFixed(1)}%</span>
          </div>
          <span className={`text-sm font-medium w-20 text-right ${
            calculatePercentDiff(metrics1.nugget_f1, metrics2.nugget_f1) > 0 
              ? 'text-green-600' : calculatePercentDiff(metrics1.nugget_f1, metrics2.nugget_f1) < 0 
              ? 'text-red-600' : 'text-gray-600'
          }`}>
            {calculatePercentDiff(metrics1.nugget_f1, metrics2.nugget_f1) > 0 ? '+' : ''}
            {calculatePercentDiff(metrics1.nugget_f1, metrics2.nugget_f1).toFixed(1)}%
          </span>
        </div>
      </div>
    </div>
  );
};

interface ComparisonRankingMetricsProps {
  metrics1: any;
  metrics2: any;
}

const ComparisonRankingMetrics: React.FC<ComparisonRankingMetricsProps> = ({ metrics1, metrics2 }) => {
  const calculatePercentDiff = (val1: number, val2: number) => {
    // Convert to display values (percentages) and calculate absolute difference (Run 2 - Run 1)
    const displayVal1 = val1 * 100;
    const displayVal2 = val2 * 100;
    return displayVal2 - displayVal1;
  };

  // Determine color based on performance thresholds
  const getPerformanceColor = (value: number) => {
    const percentage = value * 100;
    if (percentage >= 70) return 'text-green-600';
    if (percentage >= 50) return 'text-yellow-600';
    return 'text-red-600';
  };

  return (
    <div className="bg-gray-50 rounded-lg shadow-sm p-4 overflow-hidden">
      <h5 className="text-lg font-bold text-gray-800 mb-4">Ranking Performance</h5>
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <span className="text-sm font-medium text-gray-700 w-20">Recall@3</span>
          <div className="flex items-center flex-1 justify-center gap-4">
            <span className={`text-lg font-bold ${getPerformanceColor(metrics1["recall@3"])}`}>{(metrics1["recall@3"] * 100).toFixed(1)}%</span>
            <span className="text-xs text-gray-500">vs</span>
            <span className={`text-lg font-bold ${getPerformanceColor(metrics2["recall@3"])}`}>{(metrics2["recall@3"] * 100).toFixed(1)}%</span>
          </div>
          <span className={`text-sm font-medium w-20 text-right ${
            calculatePercentDiff(metrics1["recall@3"], metrics2["recall@3"]) > 0 
              ? 'text-green-600' : calculatePercentDiff(metrics1["recall@3"], metrics2["recall@3"]) < 0 
              ? 'text-red-600' : 'text-gray-600'
          }`}>
            {calculatePercentDiff(metrics1["recall@3"], metrics2["recall@3"]) > 0 ? '+' : ''}
            {calculatePercentDiff(metrics1["recall@3"], metrics2["recall@3"]).toFixed(1)}%
          </span>
        </div>
        <div className="flex items-center justify-between">
          <span className="text-sm font-medium text-gray-700 w-20">Recall@5</span>
          <div className="flex items-center flex-1 justify-center gap-4">
            <span className={`text-lg font-bold ${getPerformanceColor(metrics1["recall@5"])}`}>{(metrics1["recall@5"] * 100).toFixed(1)}%</span>
            <span className="text-xs text-gray-500">vs</span>
            <span className={`text-lg font-bold ${getPerformanceColor(metrics2["recall@5"])}`}>{(metrics2["recall@5"] * 100).toFixed(1)}%</span>
          </div>
          <span className={`text-sm font-medium w-20 text-right ${
            calculatePercentDiff(metrics1["recall@5"], metrics2["recall@5"]) > 0 
              ? 'text-green-600' : calculatePercentDiff(metrics1["recall@5"], metrics2["recall@5"]) < 0 
              ? 'text-red-600' : 'text-gray-600'
          }`}>
            {calculatePercentDiff(metrics1["recall@5"], metrics2["recall@5"]) > 0 ? '+' : ''}
            {calculatePercentDiff(metrics1["recall@5"], metrics2["recall@5"]).toFixed(1)}%
          </span>
        </div>
        <div className="flex items-center justify-between">
          <span className="text-sm font-medium text-gray-700 w-20">NDCG@5</span>
          <div className="flex items-center flex-1 justify-center gap-4">
            <span className={`text-lg font-bold ${getPerformanceColor(metrics1["ndcg@5"])}`}>{(metrics1["ndcg@5"] * 100).toFixed(1)}%</span>
            <span className="text-xs text-gray-500">vs</span>
            <span className={`text-lg font-bold ${getPerformanceColor(metrics2["ndcg@5"])}`}>{(metrics2["ndcg@5"] * 100).toFixed(1)}%</span>
          </div>
          <span className={`text-sm font-medium w-20 text-right ${
            calculatePercentDiff(metrics1["ndcg@5"], metrics2["ndcg@5"]) > 0 
              ? 'text-green-600' : calculatePercentDiff(metrics1["ndcg@5"], metrics2["ndcg@5"]) < 0 
              ? 'text-red-600' : 'text-gray-600'
          }`}>
            {calculatePercentDiff(metrics1["ndcg@5"], metrics2["ndcg@5"]) > 0 ? '+' : ''}
            {calculatePercentDiff(metrics1["ndcg@5"], metrics2["ndcg@5"]).toFixed(1)}%
          </span>
        </div>
      </div>
    </div>
  );
};

export default function TestResultsPage() {
  const [testData, setTestData] = useState<any>(null);
  const [selectedCategory, setSelectedCategory] = useState('all');
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedReport, setSelectedReport] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [isRunningTest, setIsRunningTest] = useState(false);
  
  // Comparison state
  const [isCompareMode, setIsCompareMode] = useState(false);
  const [selectedReports, setSelectedReports] = useState<any[]>([]);
  const [showComparison, setShowComparison] = useState(false);

  useEffect(() => {
    // Load test results data
    const loadData = async () => {
      try {
        const response = await fetch('/reports');
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        setTestData(data);
        setError(null);
      } catch (err) {
        console.error('Failed to load test data:', err);
        const errorMessage = err instanceof Error ? err.message : 'Unknown error occurred';
        setError(`Failed to connect to backend: ${errorMessage}`);
        setTestData(null);
      } finally {
        setLoading(false);
      }
    };

    loadData();
  }, []);

  // Comparison handler functions
  const handleToggleCompareMode = () => {
    setIsCompareMode(!isCompareMode);
    setSelectedReports([]);
  };

  const handleSelectForComparison = (report: any) => {
    setSelectedReports(prev => {
      const isSelected = prev.some(r => r.run_id === report.run_id);
      if (isSelected) {
        return prev.filter(r => r.run_id !== report.run_id);
      } else if (prev.length < 2) {
        return [...prev, report];
      }
      return prev;
    });
  };

  const handleCompare = () => {
    if (selectedReports.length === 2) {
      setShowComparison(true);
    }
  };

  const handleRunNewTest = async () => {
    setIsRunningTest(true);
    
    try {
      const response = await fetch('/run-tests', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        }
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const result = await response.json();
      
      // Refresh the data after a successful test run
      setTimeout(async () => {
        try {
          const response = await fetch('/reports');
          if (response.ok) {
            const data = await response.json();
            setTestData(data);
          }
        } catch (err) {
          console.error('Failed to refresh data:', err);
        }
      }, 2000);
      
    } catch (err) {
      console.error('Failed to run test:', err);
    } finally {
      setIsRunningTest(false);
    }
  };

  // Get all unique categories from the current data
  const getAvailableCategories = (report: any) => {
    if (!report?.predictions_data?.predictions) return [];
    const categories = new Set<string>();
    
    report.predictions_data.predictions.forEach((p: any) => {
      // Handle cases where category might be null, undefined, or empty
      const category = p.category || 'Uncategorized';
      categories.add(category);
    });
    
    return Array.from(categories).sort();
  };

  // Get categories for comparison (intersection of both reports)
  const getComparisonCategories = () => {
    if (!selectedReports[0]?.predictions_data?.predictions || !selectedReports[1]?.predictions_data?.predictions) return [];
    
    const cats1 = new Set<string>();
    const cats2 = new Set<string>();
    
    selectedReports[0].predictions_data.predictions.forEach((p: any) => {
      const category = p.category || 'Uncategorized';
      cats1.add(category);
    });
    
    selectedReports[1].predictions_data.predictions.forEach((p: any) => {
      const category = p.category || 'Uncategorized';
      cats2.add(category);
    });
    
    // Get intersection of categories (categories that exist in both reports)
    const intersection = Array.from(cats1).filter(cat => cats2.has(cat));
    return intersection.sort();
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-[var(--unh-white)] flex items-center justify-center">
        <div className="text-xl">Loading test results...</div>
      </div>
    );
  }

  if (error || !testData) {
    return (
      <main className="min-h-screen bg-[var(--unh-white)]">
        <header className="bg-[var(--unh-blue)] px-8 py-4 text-center shadow-md" style={{ color: '#fff' }}>
          <div className="flex items-center justify-between">
            <div className="flex items-center">
              <img 
                src="/unh.svg" 
                alt="UNH Logo" 
                className="my-6 mr-4" 
                style={{ maxWidth: '125px', height: 'auto', width: 'auto', marginTop: '24px', marginBottom: '24px' }} 
              />
              <span className="text-3xl font-bold whitespace-nowrap" style={{ fontFamily: 'Glypha, Arial, sans-serif' }}>
                Test Dashboard
              </span>
            </div>
          </div>
        </header>

        <div className="container mx-auto px-8 py-8">
          <div className="text-center">
            <div className="mb-6 bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded max-w-lg mx-auto">
              <strong>Error:</strong> {error || 'Unable to load test results'}
            </div>
          </div>
        </div>
      </main>
    );
  }

  return (
    <main className="min-h-screen bg-[var(--unh-white)] overflow-auto">
      <header className="bg-[var(--unh-blue)] px-8 py-4 text-center shadow-md sticky top-0 z-10" style={{ color: '#fff' }}>
        <div className="flex items-center justify-between">
          <div className="flex items-center">
            <img 
              src="/unh.svg" 
              alt="UNH Logo" 
              className="my-6 mr-4" 
              style={{ maxWidth: '125px', height: 'auto', width: 'auto', marginTop: '24px', marginBottom: '24px' }} 
            />
            <div className="text-left">
              <span className="text-3xl font-bold whitespace-nowrap block" style={{ fontFamily: 'Glypha, Arial, sans-serif' }}>
                Test Dashboard
              </span>
              <p className="text-blue-100 text-sm mt-1 whitespace-nowrap">View and compare automated testing results across multiple report runs</p>
            </div>
          </div>
          <div className="flex items-center space-x-4">
            {isCompareMode && selectedReports.length === 2 && (
              <button
                onClick={handleCompare}
                className="bg-green-500 text-white px-4 py-2 rounded-lg hover:bg-green-600 transition-colors font-medium"
              >
                Compare Selected
              </button>
            )}
            
            <button
              onClick={handleToggleCompareMode}
              className={`px-4 py-2 rounded-lg transition-colors font-medium ${
                isCompareMode 
                  ? 'bg-red-500 text-white hover:bg-red-600' 
                  : 'bg-gray-300 text-gray-700 hover:bg-gray-400'
              }`}
            >
              {isCompareMode ? 'Cancel Compare' : 'Compare Mode'}
            </button>
            
            {isRunningTest ? (
              <div className="flex items-center">
                <div className="animate-spin rounded-full h-6 w-6 border-2 border-white border-t-transparent mr-3"></div>
                <span className="text-sm">Running Test...</span>
              </div>
            ) : (
              <button
                onClick={handleRunNewTest}
                className="bg-white text-[var(--unh-blue)] px-6 py-2 rounded-lg hover:bg-blue-50 transition-colors font-medium"
              >
                Start Test Run
              </button>
            )}
          </div>
        </div>
      </header>

      <div className="container mx-auto px-8 py-8 pb-16">
        {/* Reports Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-8">
          {testData.test_runs && testData.test_runs.length > 0 ? (
            testData.test_runs.map((testRun: any, index: number) => {
              const isSelected = selectedReports.some(r => r.run_id === testRun.run_id);
              
              return (
                <div 
                  key={testRun.run_id} 
                  className={`bg-white rounded-lg shadow-lg border overflow-hidden transition-all ${
                    isCompareMode && isSelected 
                      ? 'border-blue-500 ring-2 ring-blue-200' 
                      : 'border-gray-200'
                  }`}
                >
                  {/* Report Card Header */}
                  <div className="bg-[var(--unh-blue)] text-white p-6 relative">
                    <div className="flex justify-between items-start">
                      <div>
                        <h2 className="text-xl font-bold mb-1">
                          Test Run #{testData.test_runs.length - index}{' '}
                          <span className="text-gray-300 font-normal">{testRun.run_id}</span>
                        </h2>
                        <p className="text-blue-100 text-xs mt-1">
                          {formatRunIdDate(testRun.run_id)}
                        </p>
                      </div>
                      <div className="text-right">
                        <div className="text-2xl font-bold">
                          {(testRun.summary.bertscore_f1 * 100).toFixed(1)}%
                        </div>
                        <div className="text-blue-100 text-xs">BERTScore F1</div>
                      </div>
                    </div>
                  </div>

                  {/* Report Summary Metrics */}
                  <div className="p-6">
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
                      <MetricCard 
                        title="SBERT Cosine" 
                        value={testRun.summary.sbert_cosine} 
                        subtitle="Sentence similarity" 
                      />
                      <MetricCard 
                        title="Recall@1" 
                        value={testRun.summary["recall@1"]} 
                        subtitle="Top result accuracy" 
                      />
                      <MetricCard 
                        title="NDCG@3" 
                        value={testRun.summary["ndcg@3"]} 
                        subtitle="Ranking quality" 
                      />
                      <MetricCard 
                        title="Nugget F1" 
                        value={testRun.summary.nugget_f1} 
                        subtitle="Nugget precision & recall" 
                      />
                    </div>

                    {/* Quick Stats */}
                    <div className="border-t border-gray-200 pt-4">
                      <div className="flex justify-between text-sm text-gray-600 mb-2">
                        <span>Questions Tested:</span>
                        <span className="font-medium">{testRun.total_questions}</span>
                      </div>
                      {testRun.predictions_data && (
                        <div className="flex justify-between text-sm text-gray-600 mb-4">
                          <span>Categories:</span>
                          <span className="font-medium">
                            {Object.keys(testRun.predictions_data.categories).length}
                          </span>
                        </div>
                      )}
                      
                      {/* Action Buttons */}
                      <div className="flex gap-2">
                        {isCompareMode ? (
                          <button 
                            onClick={() => handleSelectForComparison(testRun)}
                            disabled={!isSelected && selectedReports.length >= 2}
                            className={`flex-1 px-4 py-2 rounded-lg transition-colors text-sm font-medium ${
                              isSelected
                                ? 'bg-red-400 text-white hover:bg-red-500'
                                : selectedReports.length >= 2
                                ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
                                : 'bg-[var(--unh-blue)] text-white hover:bg-[var(--unh-accent-blue)]'
                            }`}
                          >
                            {isSelected ? 'Deselect' : 'Select for Compare'}
                          </button>
                        ) : (
                          <button 
                            onClick={() => {
                              setSelectedReport(testRun);
                            }}
                            className="flex-1 bg-[var(--unh-blue)] text-white px-4 py-2 rounded-lg hover:bg-[var(--unh-accent-blue)] transition-colors text-sm font-medium"
                          >
                            View Details
                          </button>
                        )}
                      </div>
                    </div>
                  </div>
                </div>
              );
            })
          ) : (
            <div className="col-span-full text-center py-12">
              <div className="text-gray-500 text-lg mb-4">No test runs found</div>
              <p className="text-gray-400">Click "Start Test Run" to generate your first test results</p>
            </div>
          )}
        </div>

        {/* Comparison Modal */}
        {showComparison && selectedReports.length === 2 && (
          <div className="fixed inset-0 bg-gray-800 bg-opacity-75 flex items-center justify-center p-4 z-50">
            <div className="bg-white rounded-lg max-w-6xl max-h-[90vh] overflow-hidden w-full">
              <div className="max-h-[90vh] overflow-y-auto">
                <div className="sticky top-0 bg-[var(--unh-blue)] border-b border-blue-600 p-6 z-20">
                  <div className="flex justify-between items-start">
                    <div>
                      <h2 className="text-2xl font-bold text-white mb-2">
                        Test Run Comparison
                      </h2>
                      <div className="flex gap-4 text-blue-100 text-sm">
                        <span>Test Run #{testData?.test_runs ? getTestRunDisplayNumber(testData.test_runs, selectedReports[0].run_id) : 1} ({formatRunIdDate(selectedReports[0].run_id)})</span>
                        <span>vs</span>
                        <span>Test Run #{testData?.test_runs ? getTestRunDisplayNumber(testData.test_runs, selectedReports[1].run_id) : 2} ({formatRunIdDate(selectedReports[1].run_id)})</span>
                      </div>
                    </div>
                    <button 
                      onClick={() => setShowComparison(false)}
                      className="text-blue-200 hover:text-white text-2xl font-bold"
                    >
                      ×
                    </button>
                  </div>
                </div>
              
                <div className="p-6">
                  {/* Comparison Summary Cards */}
                  <ComparisonSummary 
                    summary1={selectedReports[0].summary} 
                    summary2={selectedReports[1].summary}
                    runNumber1={testData?.test_runs ? getTestRunDisplayNumber(testData.test_runs, selectedReports[0].run_id) : 1}
                    runNumber2={testData?.test_runs ? getTestRunDisplayNumber(testData.test_runs, selectedReports[1].run_id) : 2}
                  />

                  {/* Detailed Summary Metrics with Circular Progress */}
                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8 relative z-10">
                    <ComparisonNuggetMetrics 
                      metrics1={selectedReports[0].summary} 
                      metrics2={selectedReports[1].summary} 
                    />
                    <ComparisonRankingMetrics 
                      metrics1={selectedReports[0].summary} 
                      metrics2={selectedReports[1].summary} 
                    />
                  </div>

                  {/* Predictions Section */}
                  {selectedReports[0]?.predictions_data && selectedReports[1]?.predictions_data ? (
                    haveSameGoldSet(selectedReports[0], selectedReports[1]) ? (
                      <div>
                        {/* Filters */}
                        <div className="bg-gray-50 rounded-lg p-4 mb-6">
                          <div className="flex flex-col md:flex-row gap-4">
                            <div>
                              <label className="block text-sm font-medium text-gray-700 mb-2">Filter by Category:</label>
                              <select 
                                value={selectedCategory}
                                onChange={(e) => setSelectedCategory(e.target.value)}
                                className="border border-gray-300 rounded-md px-3 py-2 focus:ring-2 focus:ring-[var(--unh-blue)] focus:border-transparent"
                              >
                                <option value="all">All Categories</option>
                                {getComparisonCategories().map(category => (
                                  <option key={category} value={category}>{category}</option>
                                ))}
                              </select>
                            </div>
                            <div className="flex-1">
                              <label className="block text-sm font-medium text-gray-700 mb-2">Search Questions:</label>
                              <input
                                type="text"
                                placeholder="Search in questions or answers..."
                                value={searchTerm}
                                onChange={(e) => setSearchTerm(e.target.value)}
                                className="w-full border border-gray-300 rounded-md px-3 py-2 focus:ring-2 focus:ring-[var(--unh-blue)] focus:border-transparent"
                              />
                            </div>
                          </div>
                        </div>

                        {/* Predictions Comparison List */}
                        <div className="space-y-6">{selectedReports[0].predictions_data?.predictions 
                            ?.filter((pred: any) => {
                              const predCategory = pred.category || 'Uncategorized';
                              const matchesCategory = selectedCategory === 'all' || predCategory === selectedCategory;
                              const matchesSearch = searchTerm === '' || 
                                pred.query.toLowerCase().includes(searchTerm.toLowerCase()) ||
                                pred.model_answer.toLowerCase().includes(searchTerm.toLowerCase()) ||
                                pred.reference_answer.toLowerCase().includes(searchTerm.toLowerCase());
                              return matchesCategory && matchesSearch;
                            })
                            .map((pred1: any, index: number) => {
                              // Find corresponding prediction in second report
                              const pred2 = selectedReports[1].predictions_data?.predictions?.find((p: any) => p.id === pred1.id);
                              if (!pred2) return null;
                              
                              const predCategory = pred1.category || 'Uncategorized';
                              
                              return (
                                <div key={pred1.id} className="border border-gray-200 rounded-lg p-6">
                                  <div className="flex justify-between items-start mb-4">
                                    <div className="flex items-center gap-3">
                                      <span className={`px-2 py-1 rounded text-sm font-bold ${getCategoryBadgeClasses(predCategory)}`}>
                                        {pred1.id}
                                      </span>
                                      <span className="text-sm text-gray-500">#{index + 1}</span>
                                    </div>
                                    <a 
                                      href={pred1.url} 
                                      target="_blank" 
                                      rel="noopener noreferrer"
                                      className="text-[var(--unh-blue)] hover:underline text-sm"
                                    >
                                      View Source
                                    </a>
                                  </div>
                                  
                                  <div className="mb-4">
                                    <h3 className="font-semibold text-lg text-gray-800 mb-2">Question:</h3>
                                    <p className="text-gray-700">{pred1.query}</p>
                                  </div>

                                  <div className="grid md:grid-cols-2 gap-6 mb-6">
                                    <div>
                                      <h4 className="font-semibold text-gray-800 mb-2">Test Run #{testData?.test_runs ? getTestRunDisplayNumber(testData.test_runs, selectedReports[0].run_id) : 1} - Model Answer:</h4>
                                      <div className="bg-blue-50 border-l-4 border-blue-400 p-4 rounded">
                                        <p className="text-gray-800">{pred1.model_answer}</p>
                                      </div>
                                    </div>
                                    
                                    <div>
                                      <h4 className="font-semibold text-gray-800 mb-2">Test Run #{testData?.test_runs ? getTestRunDisplayNumber(testData.test_runs, selectedReports[1].run_id) : 2} - Model Answer:</h4>
                                      <div className="bg-blue-50 border-l-4 border-blue-400 p-4 rounded">
                                        <p className="text-gray-800">{pred2.model_answer}</p>
                                      </div>
                                    </div>
                                  </div>

                                  <div className="mb-4">
                                    <h4 className="font-semibold text-gray-800 mb-2">Reference Answer:</h4>
                                    <div className="bg-green-50 border-l-4 border-green-400 p-4 rounded">
                                      <p className="text-gray-800">{pred1.reference_answer}</p>
                                    </div>
                                  </div>

                                  <div className="grid md:grid-cols-2 gap-6 mb-4">
                                    <div>
                                      <RetrievedDocuments 
                                        retrievedIds={pred1.retrieved_ids} 
                                        title={`Test Run #${testData?.test_runs ? getTestRunDisplayNumber(testData.test_runs, selectedReports[0].run_id) : 1} - Retrieved Documents:`}
                                      />
                                    </div>

                                    <div>
                                      <RetrievedDocuments 
                                        retrievedIds={pred2.retrieved_ids} 
                                        title={`Test Run #${testData?.test_runs ? getTestRunDisplayNumber(testData.test_runs, selectedReports[1].run_id) : 2} - Retrieved Documents:`}
                                      />
                                    </div>
                                  </div>

                                  {(pred1.nuggets && pred1.nuggets.length > 0) || (pred2.nuggets && pred2.nuggets.length > 0) && (
                                    <div className="grid md:grid-cols-2 gap-6 mb-6">
                                      <div>
                                        <h4 className="font-semibold text-gray-800 mb-2">Test Run #{testData?.test_runs ? getTestRunDisplayNumber(testData.test_runs, selectedReports[0].run_id) : 1} - Key Points:</h4>
                                        {pred1.nuggets && pred1.nuggets.length > 0 ? (
                                          <ul className="list-disc list-inside text-sm text-gray-700">
                                            {pred1.nuggets.map((nugget: string, idx: number) => (
                                              <li key={idx}>{nugget}</li>
                                            ))}
                                          </ul>
                                        ) : (
                                          <p className="text-sm text-gray-500 italic">No key points identified</p>
                                        )}
                                      </div>
                                      
                                      <div>
                                        <h4 className="font-semibold text-gray-800 mb-2">Test Run #{testData?.test_runs ? getTestRunDisplayNumber(testData.test_runs, selectedReports[1].run_id) : 2} - Key Points:</h4>
                                        {pred2.nuggets && pred2.nuggets.length > 0 ? (
                                          <ul className="list-disc list-inside text-sm text-gray-700">
                                            {pred2.nuggets.map((nugget: string, idx: number) => (
                                              <li key={idx}>{nugget}</li>
                                            ))}
                                          </ul>
                                        ) : (
                                          <p className="text-sm text-gray-500 italic">No key points identified</p>
                                        )}
                                      </div>
                                    </div>
                                  )}

                                  {/* Individual Question Metrics Comparison */}
                                  {pred1.metrics && pred2.metrics && (
                                    <div className="mt-6">
                                      <h4 className="font-semibold text-gray-800 mb-4">Individual Question Metrics Comparison</h4>
                                      
                                      <ComparisonSummary 
                                        summary1={pred1.metrics} 
                                        summary2={pred2.metrics}
                                        runNumber1={testData?.test_runs ? getTestRunDisplayNumber(testData.test_runs, selectedReports[0].run_id) : 1}
                                        runNumber2={testData?.test_runs ? getTestRunDisplayNumber(testData.test_runs, selectedReports[1].run_id) : 2}
                                      />

                                      {/* Detailed Metrics Comparison */}
                                      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 relative z-10">
                                        <ComparisonNuggetMetrics metrics1={pred1.metrics} metrics2={pred2.metrics} />
                                        <ComparisonRankingMetrics metrics1={pred1.metrics} metrics2={pred2.metrics} />
                                      </div>
                                    </div>
                                  )}
                                </div>
                              );
                            })
                            .filter(Boolean)}
                        {!selectedReports[0].predictions_data?.predictions && (
                          <div className="text-center py-8 text-gray-500">
                            <p>No detailed predictions available for comparison.</p>
                            <p className="text-sm mt-2">Only summary metrics are available.</p>
                          </div>
                        )}
                        </div>
                      </div>
                    ) : (
                      <div className="text-center py-8 text-gray-500">
                        <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-6 max-w-2xl mx-auto">
                          <div className="flex items-center justify-center mb-4">
                            <svg className="w-12 h-12 text-yellow-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L4.082 16.5c-.77.833.192 2.5 1.732 2.5z" />
                            </svg>
                          </div>
                          <h3 className="text-lg font-semibold text-gray-800 mb-3">Different Question Sets</h3>
                          <p className="text-gray-600 mb-2">
                            These test runs were conducted with different sets of questions and cannot be compared at the individual question level.
                          </p>
                          <p className="text-sm text-gray-500">
                            Only summary-level metrics comparison is available for test runs with different gold standards.
                          </p>
                        </div>
                      </div>
                    )
                  ) : null}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Selected Report Detail Modal */}
        {selectedReport && (
          <div className="fixed inset-0 bg-gray-800 bg-opacity-75 flex items-center justify-center p-4 z-50">
            <div className="bg-white rounded-lg max-w-6xl max-h-[90vh] overflow-hidden w-full">
              <div className="max-h-[90vh] overflow-y-auto">
                <div className="sticky top-0 bg-[var(--unh-blue)] border-b border-blue-600 p-6 z-20">
                  <div className="flex justify-between items-start">
                    <div>
                      <h2 className="text-2xl font-bold text-white mb-2">
                        Test Run #{(() => {
                          const runIndex = testData.test_runs.findIndex((run: any) => run.run_id === selectedReport.run_id);
                          return testData.test_runs.length - runIndex;
                        })()} <span className="text-gray-300 font-normal">{selectedReport.run_id}</span>
                      </h2>
                      <p className="text-blue-100 text-xs mt-1">{formatRunIdDate(selectedReport.run_id)}</p>
                    </div>
                    <button 
                      onClick={() => setSelectedReport(null)}
                      className="text-blue-200 hover:text-white text-2xl font-bold"
                    >
                      ×
                    </button>
                  </div>
                </div>
              
              <div className="p-6">
                {/* BERTscore Summary Cards */}
                <MetricsSummary summary={selectedReport.summary} />

                {/* Detailed Summary Metrics with Circular Progress */}
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8 relative z-10">
                  <NuggetMetrics metrics={selectedReport.summary} />
                  <RankingMetrics metrics={selectedReport.summary} />
                </div>

                {/* Predictions Section */}
                {selectedReport?.predictions_data && (
                  <div>
                    {/* Filters */}
                    <div className="bg-gray-50 rounded-lg p-4 mb-6">
                      <div className="flex flex-col md:flex-row gap-4">
                        <div>
                          <label className="block text-sm font-medium text-gray-700 mb-2">Filter by Category:</label>
                          <select 
                            value={selectedCategory}
                            onChange={(e) => setSelectedCategory(e.target.value)}
                            className="border border-gray-300 rounded-md px-3 py-2 focus:ring-2 focus:ring-[var(--unh-blue)] focus:border-transparent"
                          >
                            <option value="all">All Categories</option>
                            {getAvailableCategories(selectedReport).map(category => (
                              <option key={category} value={category}>{category}</option>
                            ))}
                          </select>
                        </div>
                        <div className="flex-1">
                          <label className="block text-sm font-medium text-gray-700 mb-2">Search Questions:</label>
                          <input
                            type="text"
                            placeholder="Search in questions or answers..."
                            value={searchTerm}
                            onChange={(e) => setSearchTerm(e.target.value)}
                            className="w-full border border-gray-300 rounded-md px-3 py-2 focus:ring-2 focus:ring-[var(--unh-blue)] focus:border-transparent"
                          />
                        </div>
                      </div>
                    </div>

                    {/* Predictions List */}
                    <div className="space-y-6">
                      {selectedReport.predictions_data?.predictions 
                          ?.filter((pred: any) => {
                            const predCategory = pred.category || 'Uncategorized';
                            const matchesCategory = selectedCategory === 'all' || predCategory === selectedCategory;
                            const matchesSearch = searchTerm === '' || 
                              pred.query.toLowerCase().includes(searchTerm.toLowerCase()) ||
                              pred.model_answer.toLowerCase().includes(searchTerm.toLowerCase()) ||
                              pred.reference_answer.toLowerCase().includes(searchTerm.toLowerCase());
                            return matchesCategory && matchesSearch;
                          })
                          .map((pred: any, index: number) => {
                            const predCategory = pred.category || 'Uncategorized';
                            
                            return (
                              <div key={pred.id} className="border border-gray-200 rounded-lg p-6">
                                <div className="flex justify-between items-start mb-4">
                                  <div className="flex items-center gap-3">
                                    <span className={`px-2 py-1 rounded text-sm font-bold ${getCategoryBadgeClasses(predCategory)}`}>
                                      {pred.id}
                                    </span>
                                    <span className="text-sm text-gray-500">#{index + 1}</span>
                                  </div>
                                  <a 
                                    href={pred.url} 
                                    target="_blank" 
                                    rel="noopener noreferrer"
                                    className="text-[var(--unh-blue)] hover:underline text-sm"
                                  >
                                    View Source
                                  </a>
                                </div>
                                
                                <div className="mb-4">
                                  <h3 className="font-semibold text-lg text-gray-800 mb-2">Question:</h3>
                                  <p className="text-gray-700">{pred.query}</p>
                                </div>

                                <div className="grid md:grid-cols-2 gap-6">
                                  <div>
                                    <h4 className="font-semibold text-gray-800 mb-2">Model Answer:</h4>
                                    <div className="bg-blue-50 border-l-4 border-blue-400 p-4 rounded">
                                      <p className="text-gray-800">{pred.model_answer}</p>
                                    </div>
                                  </div>
                                  
                                  <div>
                                    <h4 className="font-semibold text-gray-800 mb-2">Reference Answer:</h4>
                                    <div className="bg-green-50 border-l-4 border-green-400 p-4 rounded">
                                      <p className="text-gray-800">{pred.reference_answer}</p>
                                    </div>
                                  </div>
                                </div>

                                <div className="mt-4">
                                  <h4 className="font-semibold text-gray-800 mb-2">Retrieved Documents:</h4>
                                  <RetrievedDocuments retrievedIds={pred.retrieved_ids} />
                                </div>

                                {pred.nuggets && pred.nuggets.length > 0 && (
                                  <div className="mt-4">
                                    <h4 className="font-semibold text-gray-800 mb-2">Key Points:</h4>
                                    <ul className="list-disc list-inside text-sm text-gray-700">
                                      {pred.nuggets.map((nugget: string, idx: number) => (
                                        <li key={idx}>{nugget}</li>
                                      ))}
                                    </ul>
                                  </div>
                                )}

                                {/* Individual Question Metrics */}
                                {pred.metrics && (
                                  <div className="mt-6">
                                    <h4 className="font-semibold text-gray-800 mb-4">Individual Question Metrics</h4>
                                    
                                    <MetricsSummary summary={pred.metrics} />

                                    {/* Detailed Metrics with Circular Progress */}
                                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 relative z-10">
                                      <NuggetMetrics metrics={pred.metrics} />
                                      <RankingMetrics metrics={pred.metrics} />
                                    </div>
                                  </div>
                                )}
                              </div>
                            );
                          })}
                      {!selectedReport.predictions_data?.predictions && (
                        <div className="text-center py-8 text-gray-500">
                          <p>No detailed predictions available for this test run.</p>
                          <p className="text-sm mt-2">Only summary metrics are available.</p>
                        </div>
                      )}
                    </div>
                  </div>
                )}
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </main>
  );
}
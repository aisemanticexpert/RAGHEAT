import React, { useState, useEffect, useCallback } from 'react';
import {
  Container,
  Grid,
  Card,
  CardContent,
  Typography,
  Button,
  TextField,
  Box,
  LinearProgress,
  Chip,
  Alert,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Tabs,
  Tab
} from '@mui/material';
import {
  ExpandMore,
  PlayArrow,
  Stop,
  TrendingUp,
  Assessment,
  Psychology,
  AccountBalance,
  ShowChart,
  Timeline,
  Analytics
} from '@mui/icons-material';
import axios from 'axios';

const PortfolioConstructionDashboard = ({ apiUrl = 'http://localhost:8001' }) => {
  const [stocks, setStocks] = useState(['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA']);
  const [portfolioResult, setPortfolioResult] = useState(null);
  const [isConstructing, setIsConstructing] = useState(false);
  const [systemStatus, setSystemStatus] = useState(null);
  const [activeAgents, setActiveAgents] = useState([]);
  const [agentResults, setAgentResults] = useState({});
  const [selectedTab, setSelectedTab] = useState(0);
  const [error, setError] = useState(null);
  const [constructionProgress, setConstructionProgress] = useState(0);
  const [currentStage, setCurrentStage] = useState('');
  const [newStock, setNewStock] = useState('');
  const [analysisDialog, setAnalysisDialog] = useState({ open: false, type: '', data: null });

  // Check system status on component mount
  useEffect(() => {
    checkSystemStatus();
    const interval = setInterval(checkSystemStatus, 30000);
    return () => clearInterval(interval);
  }, []);

  const checkSystemStatus = async () => {
    try {
      const response = await axios.get(`${apiUrl}/system/status`);
      setSystemStatus(response.data);
    } catch (error) {
      console.error('Error checking system status:', error);
      setError('Unable to connect to portfolio system');
    }
  };

  const constructPortfolio = async () => {
    if (stocks.length === 0) {
      setError('Please add at least one stock symbol');
      return;
    }

    setIsConstructing(true);
    setError(null);
    setConstructionProgress(0);
    setCurrentStage('Initializing portfolio construction...');

    try {
      const response = await axios.post(`${apiUrl}/portfolio/construct`, {
        stocks: stocks,
        market_data: {
          risk_free_rate: 0.05,
          market_volatility: 0.15
        }
      });

      // Simulate progress updates (in real implementation, this would be WebSocket updates)
      const stages = [
        'Fundamental Analysis',
        'Sentiment Analysis', 
        'Valuation Analysis',
        'Heat Diffusion Calculation',
        'Risk Assessment',
        'Portfolio Optimization',
        'Final Validation'
      ];

      for (let i = 0; i < stages.length; i++) {
        setTimeout(() => {
          setCurrentStage(stages[i]);
          setConstructionProgress(((i + 1) / stages.length) * 100);
        }, i * 1000);
      }

      setTimeout(() => {
        setPortfolioResult(response.data);
        setIsConstructing(false);
        setConstructionProgress(100);
        setCurrentStage('Portfolio construction completed');
      }, stages.length * 1000);

    } catch (error) {
      console.error('Error constructing portfolio:', error);
      setError(error.response?.data?.detail || 'Error constructing portfolio');
      setIsConstructing(false);
      setConstructionProgress(0);
      setCurrentStage('');
    }
  };

  const runIndividualAnalysis = async (analysisType) => {
    try {
      const response = await axios.post(`${apiUrl}/analysis/${analysisType}`, {
        stocks: stocks
      });
      setAgentResults({
        ...agentResults,
        [analysisType]: response.data
      });
      setAnalysisDialog({
        open: true,
        type: analysisType,
        data: response.data
      });
    } catch (error) {
      console.error(`Error running ${analysisType} analysis:`, error);
      setError(`Error running ${analysisType} analysis`);
    }
  };

  const addStock = () => {
    if (newStock && !stocks.includes(newStock.toUpperCase())) {
      setStocks([...stocks, newStock.toUpperCase()]);
      setNewStock('');
    }
  };

  const removeStock = (stockToRemove) => {
    setStocks(stocks.filter(stock => stock !== stockToRemove));
  };

  const getAgentIcon = (agentType) => {
    switch (agentType) {
      case 'fundamental': return <AccountBalance />;
      case 'sentiment': return <Psychology />;
      case 'technical': return <ShowChart />;
      case 'heat-diffusion': return <Timeline />;
      default: return <Analytics />;
    }
  };

  const renderPortfolioSummary = () => {
    if (!portfolioResult) return null;

    const { portfolio_weights, performance_metrics, risk_analysis, agent_insights } = portfolioResult;

    return (
      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                <TrendingUp /> Portfolio Allocation
              </Typography>
              <TableContainer>
                <Table size="small">
                  <TableHead>
                    <TableRow>
                      <TableCell>Stock</TableCell>
                      <TableCell align="right">Weight</TableCell>
                      <TableCell align="right">Amount</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {Object.entries(portfolio_weights || {}).map(([stock, weight]) => (
                      <TableRow key={stock}>
                        <TableCell>{stock}</TableCell>
                        <TableCell align="right">{(weight * 100).toFixed(2)}%</TableCell>
                        <TableCell align="right">${(weight * 100000).toFixed(0)}</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                <Assessment /> Performance Metrics
              </Typography>
              <Box display="flex" flexDirection="column" gap={1}>
                <Box display="flex" justifyContent="space-between">
                  <Typography>Expected Return:</Typography>
                  <Typography color="primary">
                    {(performance_metrics?.expected_return * 100 || 0).toFixed(2)}%
                  </Typography>
                </Box>
                <Box display="flex" justifyContent="space-between">
                  <Typography>Volatility:</Typography>
                  <Typography color="secondary">
                    {(performance_metrics?.volatility * 100 || 0).toFixed(2)}%
                  </Typography>
                </Box>
                <Box display="flex" justifyContent="space-between">
                  <Typography>Sharpe Ratio:</Typography>
                  <Typography>
                    {(performance_metrics?.sharpe_ratio || 0).toFixed(3)}
                  </Typography>
                </Box>
                <Box display="flex" justifyContent="space-between">
                  <Typography>Max Drawdown:</Typography>
                  <Typography color="error">
                    {(performance_metrics?.max_drawdown * 100 || 0).toFixed(2)}%
                  </Typography>
                </Box>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Agent Insights
              </Typography>
              {Object.entries(agent_insights || {}).map(([agent, insights]) => (
                <Accordion key={agent}>
                  <AccordionSummary expandIcon={<ExpandMore />}>
                    <Box display="flex" alignItems="center" gap={1}>
                      {getAgentIcon(agent)}
                      <Typography variant="subtitle1">
                        {agent.replace('_', ' ').toUpperCase()} Agent
                      </Typography>
                    </Box>
                  </AccordionSummary>
                  <AccordionDetails>
                    <Typography variant="body2" style={{ whiteSpace: 'pre-wrap' }}>
                      {typeof insights === 'string' ? insights : JSON.stringify(insights, null, 2)}
                    </Typography>
                  </AccordionDetails>
                </Accordion>
              ))}
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    );
  };

  return (
    <Container maxWidth="xl" sx={{ mt: 2 }}>
      <Typography variant="h4" gutterBottom align="center">
        🤖 Multi-Agent Portfolio Construction System
      </Typography>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}

      {systemStatus && (
        <Card sx={{ mb: 3 }}>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              System Status
            </Typography>
            <Box display="flex" flexWrap="wrap" gap={1}>
              {systemStatus.agents?.map(agent => (
                <Chip
                  key={agent.name}
                  label={`${agent.name}: ${agent.status}`}
                  color={agent.status === 'active' ? 'success' : 'default'}
                  size="small"
                />
              ))}
            </Box>
          </CardContent>
        </Card>
      )}

      <Grid container spacing={3}>
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Stock Selection
              </Typography>
              
              <Box display="flex" gap={1} mb={2}>
                <TextField
                  label="Add Stock Symbol"
                  value={newStock}
                  onChange={(e) => setNewStock(e.target.value.toUpperCase())}
                  onKeyPress={(e) => e.key === 'Enter' && addStock()}
                  size="small"
                  fullWidth
                />
                <Button onClick={addStock} variant="outlined">
                  Add
                </Button>
              </Box>

              <Box display="flex" flexWrap="wrap" gap={1} mb={2}>
                {stocks.map(stock => (
                  <Chip
                    key={stock}
                    label={stock}
                    onDelete={() => removeStock(stock)}
                    color="primary"
                  />
                ))}
              </Box>

              <Button
                fullWidth
                variant="contained"
                size="large"
                onClick={constructPortfolio}
                disabled={isConstructing || stocks.length === 0}
                startIcon={isConstructing ? <Stop /> : <PlayArrow />}
              >
                {isConstructing ? 'Constructing...' : 'Construct Portfolio'}
              </Button>

              {isConstructing && (
                <Box mt={2}>
                  <Typography variant="body2" color="textSecondary" gutterBottom>
                    {currentStage}
                  </Typography>
                  <LinearProgress variant="determinate" value={constructionProgress} />
                </Box>
              )}
            </CardContent>
          </Card>

          <Card sx={{ mt: 2 }}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Individual Analyses
              </Typography>
              <Box display="flex" flexDirection="column" gap={1}>
                <Button
                  variant="outlined"
                  startIcon={<AccountBalance />}
                  onClick={() => runIndividualAnalysis('fundamental')}
                  disabled={stocks.length === 0}
                >
                  Fundamental Analysis
                </Button>
                <Button
                  variant="outlined"
                  startIcon={<Psychology />}
                  onClick={() => runIndividualAnalysis('sentiment')}
                  disabled={stocks.length === 0}
                >
                  Sentiment Analysis
                </Button>
                <Button
                  variant="outlined"
                  startIcon={<ShowChart />}
                  onClick={() => runIndividualAnalysis('technical')}
                  disabled={stocks.length === 0}
                >
                  Technical Analysis
                </Button>
                <Button
                  variant="outlined"
                  startIcon={<Timeline />}
                  onClick={() => runIndividualAnalysis('heat-diffusion')}
                  disabled={stocks.length === 0}
                >
                  Heat Diffusion
                </Button>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={8}>
          {portfolioResult ? (
            <Paper>
              <Tabs value={selectedTab} onChange={(e, v) => setSelectedTab(v)}>
                <Tab label="Portfolio Summary" />
                <Tab label="Detailed Analysis" />
                <Tab label="Risk Metrics" />
              </Tabs>
              <Box p={3}>
                {selectedTab === 0 && renderPortfolioSummary()}
                {selectedTab === 1 && (
                  <Typography>Detailed agent analysis will be shown here</Typography>
                )}
                {selectedTab === 2 && (
                  <Typography>Risk analysis metrics will be shown here</Typography>
                )}
              </Box>
            </Paper>
          ) : (
            <Paper>
              <Box p={4} textAlign="center">
                <Typography variant="h6" color="textSecondary" gutterBottom>
                  No portfolio constructed yet
                </Typography>
                <Typography color="textSecondary">
                  Select stocks and click "Construct Portfolio" to begin multi-agent analysis
                </Typography>
              </Box>
            </Paper>
          )}
        </Grid>
      </Grid>

      <Dialog
        open={analysisDialog.open}
        onClose={() => setAnalysisDialog({ open: false, type: '', data: null })}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>
          {analysisDialog.type?.toUpperCase()} Analysis Results
        </DialogTitle>
        <DialogContent>
          <pre style={{ whiteSpace: 'pre-wrap', fontSize: '0.9rem' }}>
            {JSON.stringify(analysisDialog.data, null, 2)}
          </pre>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setAnalysisDialog({ open: false, type: '', data: null })}>
            Close
          </Button>
        </DialogActions>
      </Dialog>
    </Container>
  );
};

export default PortfolioConstructionDashboard;
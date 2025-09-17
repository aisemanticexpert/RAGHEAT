# 🚀 RAGHeat Testing Framework v2.0.0

Professional automated testing framework for the RAGHeat Multi-Agent Portfolio Construction System.

## 📊 Overview

This comprehensive testing framework provides enterprise-grade automation for the RAGHeat application, featuring:

- **45 Test Cases** across Smoke, Sanity, and Regression test suites
- **Professional Email Reporting** with HTML templates and responsive design
- **Docker Containerization** for consistent testing environments
- **Jenkins CI/CD Integration** for automated pipeline execution
- **XPath-based UI Automation** with comprehensive element mapping
- **Multi-Browser Support** with Chrome (headless/visible modes)
- **Performance Testing** and edge case validation
- **Mathematical Data Integrity** verification

## 🎯 Test Suite Structure

### 🔥 Smoke Tests (15 tests)
Critical path validation ensuring core functionality:
- Application startup and health checks
- Portfolio construction API endpoints
- Dashboard navigation and UI responsiveness
- API endpoint accessibility

### 📋 Sanity Tests (15 tests)
Functional validation of key features:
- Complete portfolio construction workflows
- Dashboard component interactions
- API functionality and data integrity
- Multi-agent system coordination

### 🔄 Regression Tests (15 tests)
Comprehensive validation for stability:
- Performance and stress testing
- Edge cases and boundary conditions
- Data integrity and mathematical correctness
- Cross-component integration

## 🏗️ Architecture

```
ragheat_testing_framework/
├── tests/                          # Test suites
│   ├── smoke/                      # Critical path tests
│   ├── sanity/                     # Functional tests
│   └── regression/                 # Comprehensive tests
├── framework/                      # Core framework
│   ├── base_test.py               # Base test class
│   ├── page_objects/              # Page object models
│   ├── utilities/                 # Utilities and helpers
│   │   ├── api_client.py         # API testing client
│   │   ├── email_templates.py    # Professional email templates
│   │   └── enhanced_email_reporter.py # Email reporting engine
│   └── config/                    # Configuration
│       └── ui_elements.json      # XPath mappings
├── jenkins/                       # CI/CD configuration
│   └── Jenkinsfile               # Pipeline definition
├── reports/                       # Test execution reports
├── screenshots/                   # UI test evidence
├── docker-compose.yml            # Container orchestration
├── Dockerfile                    # Testing environment image
└── deploy_automation_framework.sh # Deployment script
```

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose
- Python 3.11+
- Chrome browser (for local execution)

### 1. Deploy the Framework
```bash
./deploy_automation_framework.sh
```

### 2. Run Tests

#### Local Execution
```bash
# Smoke tests (critical path)
python3 run_tests.py --suite smoke --headless true

# All test suites
python3 run_tests.py --suite all --headless true

# Visible browser testing
python3 run_tests.py --suite smoke --headless false
```

#### Docker Execution
```bash
# Run complete test suite
docker-compose up ragheat-testing

# Start Jenkins for CI/CD
docker-compose up jenkins

# Start test report server
docker-compose up test-reporter
```

## 🌐 Access Points

- **Jenkins UI**: http://localhost:8080
- **Test Reports**: http://localhost:8888/reports/
- **Selenium Grid**: http://localhost:4444

## 📧 Email Reporting

The framework includes professional email reporting with:

- **Modern HTML Templates**: Responsive design with professional styling
- **Success/Failure Templates**: Different templates based on test outcomes
- **Comprehensive Statistics**: Test metrics, performance data, and summaries
- **Attachment Support**: Screenshots, logs, and HTML reports
- **Mobile-Friendly**: Responsive design for all devices

### Email Configuration
```bash
export SENDER_EMAIL="your-email@gmail.com"
export SENDER_PASSWORD="your-app-password"
export EMAIL_MOCK_MODE="false"  # Set to true for testing
```

## 🎨 UI Element Automation

The framework uses comprehensive XPath mappings for UI automation:

```json
{
  "dashboard_buttons": {
    "revolutionary_dashboard": "//button[contains(text(), 'REVOLUTIONARY DASHBOARD')]",
    "portfolio_ai": "//button[contains(text(), 'PORTFOLIO AI')]"
  },
  "portfolio_construction": {
    "stock_input": "//input[@placeholder='Enter stock symbol']",
    "construct_button": "//button[contains(text(), 'Construct Portfolio')]"
  }
}
```

## 🔧 Configuration

### Environment Variables
- `HEADLESS`: Browser mode (true/false)
- `RAGHEAT_FRONTEND_URL`: Frontend application URL
- `RAGHEAT_API_URL`: Backend API URL
- `EMAIL_MOCK_MODE`: Email testing mode

### Test Execution Options
```bash
# Available test suites
--suite smoke      # Critical path tests
--suite sanity     # Functional validation
--suite regression # Comprehensive testing
--suite all        # Complete test suite

# Environment options
--env local        # localhost:3000, localhost:8000
--env staging      # staging.semanticdataservices.com
--env production   # www.semanticdataservices.com

# Browser options
--headless true    # Headless Chrome
--headless false   # Visible browser
```

## 🏭 Jenkins CI/CD Pipeline

The framework includes a complete Jenkins pipeline:

1. **Environment Setup**: Clean environment preparation
2. **Service Startup**: RAGHeat application deployment
3. **Smoke Tests**: Critical path validation (fails fast)
4. **Sanity Tests**: Functional verification
5. **Regression Tests**: Comprehensive validation
6. **Report Generation**: HTML reports and email notifications
7. **Artifact Archival**: Screenshots, logs, and reports

### Pipeline Triggers
- **Code commits**: Automatic testing on push
- **Scheduled runs**: Nightly regression testing
- **Manual execution**: On-demand test runs

## 📊 Test Reports

The framework generates comprehensive reports:

- **HTML Reports**: Detailed test execution results with screenshots
- **Email Reports**: Professional summaries sent to stakeholders
- **JSON Summaries**: Machine-readable test data
- **Performance Metrics**: Load times and response measurements

## 🔒 Security & Best Practices

- **Environment Variables**: Secure credential management
- **Mock Mode**: Safe testing without external dependencies
- **Container Isolation**: Secure Docker-based execution
- **Access Controls**: Jenkins security configuration

## 🛠️ Development

### Adding New Tests
1. Create test file in appropriate suite directory
2. Extend `BaseTest` class
3. Use XPath mappings from `ui_elements.json`
4. Follow naming convention: `test_*.py`

### Extending Email Templates
1. Add new template to `email_templates.py`
2. Update `enhanced_email_reporter.py`
3. Test with mock mode enabled

### Custom Page Objects
1. Create new page object in `page_objects/`
2. Follow existing patterns
3. Use XPath selectors from configuration

## 📈 Performance Benchmarks

The framework is optimized for:
- **Test Execution**: < 5 minutes for complete suite
- **Email Delivery**: < 10 seconds for report generation
- **Container Startup**: < 30 seconds for full environment
- **Report Generation**: < 5 seconds for HTML reports

## 🆘 Troubleshooting

### Common Issues
1. **Docker Build Failures**: Check disk space and permissions
2. **Browser Launch Issues**: Verify Chrome installation
3. **Network Timeouts**: Adjust timeout settings in configuration
4. **Email Delivery**: Verify SMTP credentials and settings

### Debug Mode
```bash
# Enable verbose logging
python3 run_tests.py --suite smoke -v -s

# Generate test preview
python3 -c "from framework.utilities.enhanced_email_reporter import EnhancedEmailReporter; reporter = EnhancedEmailReporter(); reporter.create_html_report_preview({'total': 5, 'passed': 4, 'failed': 1, 'skipped': 0}, 'preview.html')"
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **RAGHeat Team**: Core application development
- **Selenium Community**: Web automation framework
- **Docker Community**: Containerization platform
- **Jenkins Community**: CI/CD automation

## 📞 Support

For support and questions:
- **Email**: semanticraj@gmail.com
- **Documentation**: [SemanticDataServices.com](https://www.semanticdataservices.com)
- **Issues**: GitHub Issues tracker

---

**🎉 RAGHeat Testing Framework v2.0.0** - Professional automated testing for semantic data services.

*Built with ❤️ for reliable, scalable, and maintainable test automation.*
def pytest_report_teststatus(report, config):
    if report.when == "call":
        if report.passed: return "passed", "", ""
        if report.failed: return "failed", "F", ""

def pytest_sessionstart(session):
    tr = session.config.pluginmanager.getplugin("terminalreporter")
    if tr is None: return
    if hasattr(tr, '_write_progress_information_filling_space'):
        tr._write_progress_information_filling_space = lambda: None
    if hasattr(tr, 'summary_stats'):
        orig = tr.summary_stats
        def _summary_stats():
            if tr.stats.get("failed"): orig()
        tr.summary_stats = _summary_stats

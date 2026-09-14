document.addEventListener('DOMContentLoaded', function () {
  const selectedTab = localStorage.getItem('selectedTab');
  if (selectedTab) {
    const tabTrigger = document.getElementById(selectedTab);
    if (tabTrigger) {
      new bootstrap.Tab(tabTrigger).show();
    }
  }

  document.querySelectorAll('button[data-bs-toggle="tab"]').forEach(tab => {
    tab.addEventListener('shown.bs.tab', function (e) {
      localStorage.setItem('selectedTab', e.target.id);
    });
  });
});

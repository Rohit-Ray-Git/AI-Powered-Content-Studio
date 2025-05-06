import React from 'react';

// Heroicons SVGs (inline for demo; in production, use a component or import)
const icons = {
  dashboard: (
    <svg className="w-5 h-5 mr-3" fill="none" stroke="currentColor" strokeWidth="1.5" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" d="M3 13.5V6.75A2.25 2.25 0 015.25 4.5h13.5A2.25 2.25 0 0121 6.75v6.75M3 13.5v3.75A2.25 2.25 0 005.25 19.5h13.5A2.25 2.25 0 0021 17.25V13.5M3 13.5h18" /></svg>
  ),
  edit: (
    <svg className="w-5 h-5 mr-3" fill="none" stroke="currentColor" strokeWidth="1.5" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" d="M16.862 4.487a2.1 2.1 0 112.97 2.97L7.5 19.79l-4 1 1-4 13.362-13.303z" /></svg>
  ),
  history: (
    <svg className="w-5 h-5 mr-3" fill="none" stroke="currentColor" strokeWidth="1.5" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>
  ),
  settings: (
    <svg className="w-5 h-5 mr-3" fill="none" stroke="currentColor" strokeWidth="1.5" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.573-1.065z" /></svg>
  ),
};

const navLinks = [
  { name: 'Dashboard', icon: icons.dashboard, active: true },
  { name: 'Content Generation', icon: icons.edit, active: false },
  { name: 'History', icon: icons.history, active: false },
  { name: 'Settings', icon: icons.settings, active: false },
];

export default function DashboardPage() {
  return (
    <div className="flex min-h-screen bg-black text-gray-100">
      {/* Sidebar */}
      <aside className="w-64 bg-black shadow-md flex flex-col">
        <div className="h-16 flex items-center justify-center border-b border-gray-800">
          {/* Logo/Icon */}
          <span className="flex items-center gap-2 text-xl font-bold text-blue-400">
            <svg className="w-7 h-7 text-blue-400" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24"><circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="2" /><path d="M8 12l2 2 4-4" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" /></svg>
            AI Content Studio
          </span>
        </div>
        <nav className="flex-1 py-4">
          <ul className="space-y-1">
            {navLinks.map((link) => (
              <li key={link.name}>
                <a
                  href="#"
                  className={`flex items-center px-6 py-2 rounded transition font-medium ${
                    link.active
                      ? 'bg-blue-950/60 text-blue-400 border-l-4 border-blue-500 shadow'
                      : 'text-gray-200 hover:bg-blue-900/30 hover:text-blue-400'
                  }`}
                >
                  {link.icon}
                  {link.name}
                </a>
              </li>
            ))}
          </ul>
        </nav>
        <div className="border-t border-gray-800 mx-4" />
        {/* User Profile Section */}
        <div className="flex items-center gap-3 p-6">
          <img
            src="https://randomuser.me/api/portraits/men/32.jpg"
            alt="User Avatar"
            className="w-10 h-10 rounded-full border-2 border-blue-400 object-cover"
          />
          <div>
            <div className="font-semibold text-gray-100">John Doe</div>
            <div className="text-xs text-gray-500">Content Strategist</div>
          </div>
        </div>
        <div className="p-6 pt-0 text-xs text-gray-500">&copy; 2024 AI Content Studio</div>
      </aside>
      {/* Main Content */}
      <main className="flex-1 p-8">
        <header className="mb-8 flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-gray-100">Dashboard</h1>
            <p className="text-gray-400 mt-2">Welcome to your AI-powered content creation dashboard.</p>
          </div>
          {/* Dark mode toggle placeholder */}
          <button className="bg-gray-900 hover:bg-gray-800 text-gray-200 px-4 py-2 rounded shadow flex items-center gap-2">
            <svg className="w-5 h-5" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" d="M21.752 15.002A9.718 9.718 0 0112 21.75c-5.385 0-9.75-4.365-9.75-9.75 0-4.136 2.664-7.64 6.398-9.09.513-.2 1.07-.03 1.385.43.316.46.23 1.09-.2 1.42A7.5 7.5 0 1019.5 12c0-.41.25-.78.64-.93.39-.15.84-.02 1.08.32.36.53.55 1.16.53 1.61z" /></svg>
            Dark Mode
          </button>
        </header>
        {/* Dashboard Cards */}
        <section className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
          <div className="bg-black border border-gray-800 rounded-lg shadow p-6 flex flex-col items-start">
            <div className="text-sm text-gray-400 mb-2">Quick Action</div>
            <div className="text-xl font-bold text-blue-400 mb-4">Generate Content</div>
            <button className="mt-auto bg-blue-700 hover:bg-blue-800 text-white px-4 py-2 rounded">Start</button>
          </div>
          <div className="bg-black border border-gray-800 rounded-lg shadow p-6 flex flex-col items-start">
            <div className="text-sm text-gray-400 mb-2">Recent Activity</div>
            <div className="text-xl font-bold text-green-400 mb-4">3 New Posts</div>
            <button className="mt-auto bg-green-700 hover:bg-green-800 text-white px-4 py-2 rounded">View</button>
          </div>
          <div className="bg-black border border-gray-800 rounded-lg shadow p-6 flex flex-col items-start">
            <div className="text-sm text-gray-400 mb-2">Settings</div>
            <div className="text-xl font-bold text-yellow-400 mb-4">Customize</div>
            <button className="mt-auto bg-yellow-700 hover:bg-yellow-800 text-white px-4 py-2 rounded">Edit</button>
          </div>
        </section>
        {/* Placeholder for more dashboard content */}
        <section className="bg-black rounded-lg shadow p-8 min-h-[200px] flex items-center justify-center border border-gray-800">
          <span className="text-gray-500 text-lg">[ More dashboard content goes here ]</span>
        </section>
      </main>
    </div>
  );
}

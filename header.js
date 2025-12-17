class CustomHeader extends HTMLElement {
    connectedCallback() {
        this.attachShadow({ mode: 'open' });
        this.shadowRoot.innerHTML = `
            <style>
                .logo-text {
                    background: linear-gradient(90deg, #7cb342 0%, #ffb300 100%);
                    -webkit-background-clip: text;
                    background-clip: text;
                    -webkit-text-fill-color: transparent;
                }
            </style>
            <header class="flex justify-between items-center">
                <div class="flex items-center">
                    <div class="w-10 h-10 rounded-full bg-primary-500 flex items-center justify-center text-white mr-3">
                        <i data-feather="eye-off"></i>
                    </div>
                    <h1 class="text-2xl font-bold logo-text">BrailleVision</h1>
                </div>
                <nav class="flex items-center space-x-6">
                    <a href="#" class="text-gray-700 hover:text-primary-700 transition-colors flex items-center">
                        <i data-feather="home" class="mr-1"></i> Home
                    </a>
                    <a href="#" class="text-gray-700 hover:text-primary-700 transition-colors flex items-center">
                        <i data-feather="book" class="mr-1"></i> Guide
                    </a>
                    <a href="#" class="text-gray-700 hover:text-primary-700 transition-colors flex items-center">
                        <i data-feather="user" class="mr-1"></i> Account
                    </a>
                </nav>
            </header>
        `;
    }
}
customElements.define('custom-header', CustomHeader);
class HamburgerMenu extends HTMLElement {
    constructor() {
        super()
    }

    connectedCallback() {

        this.innerHTML = `
        <nav role="navigation">
            <div id="menuToggle">
                <link rel="stylesheet" href="/components/hamburger.css" />
                <input type="checkbox" />
                <span></span>
                <span></span>
                <span></span>
                <ul id="menu">
                    <div style="width:100%;margin-top:-85px;margin-left:20px;display:flex;flex-direction:row;align-items:center;justify-content:flex-end">
                        <a href="/profile">
                            <h4 id="logged_as" style=";padding-top:5px;margin-right:20px"></h4>
                        </a>
                        <button id="login_btn" class="login_btn"> 
                            <h3 style="margin:0px;margin-right:15px;color:white">Log in </h3> 
                            <img src="/assets/google.svg" style="width:30px;"/>
                        </button>
                    </div>
                    <a href="#"><li>Home</li></a>
                    <a href="#"><li>About</li></a>
                    <a href="#"><li>Info</li></a>
                    <a href="#"><li>Contact</li></a>
                    <a href="https://erikterwan.com/" target="_blank"><li>Show me more</li></a>
                </ul>
            </div>
        </nav>
        `
    }
}

customElements.define("hamburger-menu", HamburgerMenu)
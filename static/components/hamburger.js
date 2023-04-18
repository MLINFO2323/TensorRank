class HamburgerMenu extends HTMLElement {
    constructor() {
        super()
    }

    connectedCallback() {

        this.innerHTML = `
        <div class="hamburger-menu">
        <input id="menu__toggle" type="checkbox" />
        <label class="menu__btn" for="menu__toggle">
        <span></span>
        </label>
        
        <ul class="menu__box">
            <li>
                <button id="login_btn" class="login_btn">
                    <h3 class="login_h3">
                        Login
                    </h3>
                    <img width=30px src="./assets/google.svg"></img>
                </button>
            </li>
            <li>
            <h5 id="logged_as" style="margin-right:20px;margin-top:0px">
            </h5>
            </li>
        </ul>
        </div>
        `
    }
}

customElements.define("hamburger-menu", HamburgerMenu)
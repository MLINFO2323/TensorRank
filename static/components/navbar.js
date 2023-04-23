class Navbar extends HTMLElement {
    static get observedAttributes() {
        return ["hide-login"]
    }
    constructor() {
        super()
    }
    connectedCallback() {

        this.innerHTML = `

        <div
            style="background: #2b2b2b;display: flex;flex-direction: row;padding:0px;height: 120px;justify-content: space-between;">
            <hamburger-menu></hamburger-menu>
            <div style="display: flex;flex-direction: row;align-items: center;">
                <a href="/"><h1>TensorRank</h1></a>
                <img src="/assets/tf.svg" height="70px" style="margin:0px 20px" alt="" />
            </div>
        </div>
        `
    }

    attributeChangedCallback(attrName, oldVal, newVal) {
        if (attrName == "hide-login") {
            this.querySelector("hamburger-menu").setAttribute("hide-login", newVal)
        }
    }
}

customElements.define("nav-bar", Navbar)
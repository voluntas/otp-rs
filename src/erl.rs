#[derive(Debug, Clone)]
pub struct Module {
    pub name: String,
    pub exports: Vec<ExportSpec>,
    pub functions: Vec<Function>,
}

#[derive(Debug, Clone)]
pub struct ExportSpec {
    pub name: String,
    pub arity: usize,
}

#[derive(Debug, Clone)]
pub struct Function {
    pub name: String,
    pub params: Vec<String>,
    pub body: Expr,
}

#[derive(Debug, Clone)]
pub enum Expr {
    Atom(String),
    Integer(i64),
    Var(String),
    Add(Box<Expr>, Box<Expr>),
    Sub(Box<Expr>, Box<Expr>),
    Mul(Box<Expr>, Box<Expr>),
    Call {
        module: Option<String>,
        function: String,
        args: Vec<Expr>,
    },
}

#[derive(Debug, Clone, PartialEq)]
enum Token {
    Atom(String),
    Var(String),
    Integer(i64),
    Dash,
    Arrow,
    LParen,
    RParen,
    LBracket,
    RBracket,
    Comma,
    Dot,
    Slash,
    Plus,
    Star,
    Colon,
}

fn tokenize(input: &str) -> Result<Vec<Token>, String> {
    let mut tokens = Vec::new();
    let mut chars = input.chars().peekable();

    while let Some(&ch) = chars.peek() {
        if ch.is_whitespace() {
            chars.next();
            continue;
        }
        if ch == '%' {
            while let Some(c) = chars.next() {
                if c == '\n' {
                    break;
                }
            }
            continue;
        }

        match ch {
            '-' => {
                chars.next();
                if chars.peek() == Some(&'>') {
                    chars.next();
                    tokens.push(Token::Arrow);
                } else {
                    tokens.push(Token::Dash);
                }
            }
            '(' => {
                chars.next();
                tokens.push(Token::LParen);
            }
            ')' => {
                chars.next();
                tokens.push(Token::RParen);
            }
            '[' => {
                chars.next();
                tokens.push(Token::LBracket);
            }
            ']' => {
                chars.next();
                tokens.push(Token::RBracket);
            }
            ',' => {
                chars.next();
                tokens.push(Token::Comma);
            }
            '.' => {
                chars.next();
                tokens.push(Token::Dot);
            }
            '/' => {
                chars.next();
                tokens.push(Token::Slash);
            }
            '+' => {
                chars.next();
                tokens.push(Token::Plus);
            }
            '*' => {
                chars.next();
                tokens.push(Token::Star);
            }
            ':' => {
                chars.next();
                tokens.push(Token::Colon);
            }
            '0'..='9' => {
                let mut value: i64 = 0;
                while let Some(&digit) = chars.peek() {
                    if digit.is_ascii_digit() {
                        chars.next();
                        value = value
                            .checked_mul(10)
                            .and_then(|v| v.checked_add((digit as u8 - b'0') as i64))
                            .ok_or_else(|| "integer overflow".to_string())?;
                    } else {
                        break;
                    }
                }
                tokens.push(Token::Integer(value));
            }
            _ => {
                if ch.is_ascii_alphabetic() || ch == '_' {
                    let mut ident = String::new();
                    while let Some(&c) = chars.peek() {
                        if c.is_ascii_alphanumeric() || c == '_' || c == '@' {
                            ident.push(c);
                            chars.next();
                        } else {
                            break;
                        }
                    }
                    if ident.is_empty() {
                        return Err("empty identifier".to_string());
                    }
                    let first = ident.chars().next().unwrap();
                    if first.is_ascii_uppercase() || first == '_' {
                        tokens.push(Token::Var(ident));
                    } else {
                        tokens.push(Token::Atom(ident));
                    }
                } else {
                    return Err(format!("unexpected character '{}'", ch));
                }
            }
        }
    }

    Ok(tokens)
}

struct Parser {
    tokens: Vec<Token>,
    pos: usize,
}

impl Parser {
    fn new(tokens: Vec<Token>) -> Self {
        Self { tokens, pos: 0 }
    }

    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.pos)
    }

    fn next(&mut self) -> Option<Token> {
        let tok = self.tokens.get(self.pos).cloned();
        if tok.is_some() {
            self.pos += 1;
        }
        tok
    }

    fn eof(&self) -> bool {
        self.pos >= self.tokens.len()
    }

    fn expect(&mut self, expected: Token) -> Result<(), String> {
        match self.next() {
            Some(tok) if tok == expected => Ok(()),
            Some(tok) => Err(format!("unexpected token {:?}", tok)),
            None => Err("unexpected end of input".to_string()),
        }
    }

    fn expect_atom(&mut self) -> Result<String, String> {
        match self.next() {
            Some(Token::Atom(name)) => Ok(name),
            Some(tok) => Err(format!("expected atom, got {:?}", tok)),
            None => Err("unexpected end of input".to_string()),
        }
    }

    fn expect_var(&mut self) -> Result<String, String> {
        match self.next() {
            Some(Token::Var(name)) => Ok(name),
            Some(tok) => Err(format!("expected var, got {:?}", tok)),
            None => Err("unexpected end of input".to_string()),
        }
    }

    fn expect_integer(&mut self) -> Result<i64, String> {
        match self.next() {
            Some(Token::Integer(value)) => Ok(value),
            Some(tok) => Err(format!("expected integer, got {:?}", tok)),
            None => Err("unexpected end of input".to_string()),
        }
    }

    fn skip_to_dot(&mut self) {
        while let Some(tok) = self.next() {
            if tok == Token::Dot {
                break;
            }
        }
    }

    fn parse_module(&mut self) -> Result<Module, String> {
        let mut module_name: Option<String> = None;
        let mut exports: Vec<ExportSpec> = Vec::new();
        let mut functions: Vec<Function> = Vec::new();

        while !self.eof() {
            match self.peek() {
                Some(Token::Dash) => {
                    self.next();
                    let attr = self.expect_atom()?;
                    match attr.as_str() {
                        "module" => {
                            self.expect(Token::LParen)?;
                            let name = self.expect_atom()?;
                            self.expect(Token::RParen)?;
                            self.expect(Token::Dot)?;
                            module_name = Some(name);
                        }
                        "export" => {
                            self.expect(Token::LParen)?;
                            self.expect(Token::LBracket)?;
                            loop {
                                let name = self.expect_atom()?;
                                self.expect(Token::Slash)?;
                                let arity = self.expect_integer()? as usize;
                                exports.push(ExportSpec { name, arity });
                                match self.peek() {
                                    Some(Token::Comma) => {
                                        self.next();
                                        continue;
                                    }
                                    Some(Token::RBracket) => break,
                                    other => {
                                        return Err(format!(
                                            "unexpected token in export list: {:?}",
                                            other
                                        ))
                                    }
                                }
                            }
                            self.expect(Token::RBracket)?;
                            self.expect(Token::RParen)?;
                            self.expect(Token::Dot)?;
                        }
                        _ => {
                            self.skip_to_dot();
                        }
                    }
                }
                Some(_) => {
                    functions.push(self.parse_function()?);
                }
                None => break,
            }
        }

        let name = module_name.ok_or_else(|| "missing -module attribute".to_string())?;
        Ok(Module {
            name,
            exports,
            functions,
        })
    }

    fn parse_function(&mut self) -> Result<Function, String> {
        let name = self.expect_atom()?;
        self.expect(Token::LParen)?;
        let mut params = Vec::new();
        match self.peek() {
            Some(Token::RParen) => {
                self.next();
            }
            _ => {
                loop {
                    let param = self.expect_var()?;
                    params.push(param);
                    match self.peek() {
                        Some(Token::Comma) => {
                            self.next();
                            continue;
                        }
                        Some(Token::RParen) => {
                            self.next();
                            break;
                        }
                        other => {
                            return Err(format!("unexpected token in params: {:?}", other));
                        }
                    }
                }
            }
        }
        self.expect(Token::Arrow)?;
        let body = self.parse_expr()?;
        self.expect(Token::Dot)?;
        Ok(Function { name, params, body })
    }

    fn parse_expr(&mut self) -> Result<Expr, String> {
        self.parse_sum()
    }

    fn parse_sum(&mut self) -> Result<Expr, String> {
        let mut expr = self.parse_product()?;
        loop {
            match self.peek() {
                Some(Token::Plus) => {
                    self.next();
                    let rhs = self.parse_product()?;
                    expr = Expr::Add(Box::new(expr), Box::new(rhs));
                }
                Some(Token::Dash) => {
                    self.next();
                    let rhs = self.parse_product()?;
                    expr = Expr::Sub(Box::new(expr), Box::new(rhs));
                }
                _ => break,
            }
        }
        Ok(expr)
    }

    fn parse_product(&mut self) -> Result<Expr, String> {
        let mut expr = self.parse_factor()?;
        loop {
            match self.peek() {
                Some(Token::Star) => {
                    self.next();
                    let rhs = self.parse_factor()?;
                    expr = Expr::Mul(Box::new(expr), Box::new(rhs));
                }
                _ => break,
            }
        }
        Ok(expr)
    }

    fn parse_factor(&mut self) -> Result<Expr, String> {
        match self.peek() {
            Some(Token::Dash) => {
                self.next();
                let expr = self.parse_factor()?;
                Ok(Expr::Sub(Box::new(Expr::Integer(0)), Box::new(expr)))
            }
            _ => self.parse_term(),
        }
    }

    fn parse_term(&mut self) -> Result<Expr, String> {
        match self.next() {
            Some(Token::Atom(name)) => match self.peek() {
                Some(Token::Colon) => {
                    self.next();
                    let func = self.expect_atom()?;
                    let args = self.parse_call_args()?;
                    Ok(Expr::Call {
                        module: Some(name),
                        function: func,
                        args,
                    })
                }
                Some(Token::LParen) => {
                    let args = self.parse_call_args()?;
                    Ok(Expr::Call {
                        module: None,
                        function: name,
                        args,
                    })
                }
                _ => Ok(Expr::Atom(name)),
            },
            Some(Token::Var(name)) => Ok(Expr::Var(name)),
            Some(Token::Integer(value)) => Ok(Expr::Integer(value)),
            Some(Token::LParen) => {
                let expr = self.parse_expr()?;
                self.expect(Token::RParen)?;
                Ok(expr)
            }
            Some(tok) => Err(format!("unexpected term token {:?}", tok)),
            None => Err("unexpected end of input".to_string()),
        }
    }

    fn parse_call_args(&mut self) -> Result<Vec<Expr>, String> {
        self.expect(Token::LParen)?;
        let mut args = Vec::new();
        match self.peek() {
            Some(Token::RParen) => {
                self.next();
                return Ok(args);
            }
            _ => {}
        }
        loop {
            let expr = self.parse_expr()?;
            args.push(expr);
            match self.peek() {
                Some(Token::Comma) => {
                    self.next();
                    continue;
                }
                Some(Token::RParen) => {
                    self.next();
                    break;
                }
                other => {
                    return Err(format!("unexpected token in call args: {:?}", other));
                }
            }
        }
        Ok(args)
    }
}

pub fn parse_module(contents: &str) -> Result<Module, String> {
    let tokens = tokenize(contents)?;
    let mut parser = Parser::new(tokens);
    let module = parser.parse_module()?;
    if !parser.eof() {
        return Err("unexpected tokens after module".to_string());
    }
    Ok(module)
}
